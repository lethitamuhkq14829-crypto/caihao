"""Read archived matched samples; add common-support diagnostics without rematching.

No new remote sensing data or spatial geometry is created. Existing outputs remain
read-only. Bootstrap matches the manuscript's 10 km consecutive chainage blocks.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'output/tables/SecondPaper_Complete_20261109'
EXT = ROOT / 'output/tables/SecondPaper_Extension_20261109'
OUT = ROOT / 'output/tables/SecondPaper_Expansion_20261113'
SEED, REPS, BLOCK = 20261113, 2000, 10
OUT.mkdir(parents=True, exist_ok=True)
inputs = []


def read(path, **kwargs):
    inputs.append(path)
    return pd.read_csv(path, **kwargs)


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.10g')


def unit_boot(values, units):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    blocks, ix = np.unique(np.asarray(units) // BLOCK, return_inverse=True)
    sums = np.zeros((len(blocks), values.shape[1]))
    np.add.at(sums, ix, values)
    n = np.bincount(ix)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(blocks), (REPS, len(blocks)))
    boot = sums[picks].sum(axis=1) / n[picks].sum(axis=1)[:, None]
    return values.mean(axis=0), np.quantile(boot, [.025, .975], axis=0), len(blocks)


def distance_checks():
    d = read(EXT / 'distance_segment_effects.csv')
    assert set(d.type) == {'Forest', 'Cropland', 'Other vegetation'}
    assert not d.duplicated(['analysis', 'route', 'unit', 'type']).any()
    d['weighted'] = d.n * d.operation
    a = d.groupby(['analysis', 'route', 'unit']).agg(weighted=('weighted', 'sum'), n=('n', 'sum')).reset_index()
    a['operation'] = a.weighted / a.n
    bands = ['0_30', '30_60', '60_90', '90_150', '150_300', '300_500']
    rows, records = [], []
    for route in ['Russia', 'China']:
        for ref in [500, 1000]:
            cols = [f'distance_{band}_ref{ref}' for band in bands]
            wide = a[a.route == route].pivot(index='unit', columns='analysis', values='operation')[cols].dropna()
            mean, ci, nblocks = unit_boot(wide.to_numpy(), wide.index.to_numpy())
            for j, name in enumerate(cols):
                s = a[(a.route == route) & (a.analysis == name) & a.unit.isin(wide.index)].copy()
                assert len(s) == len(wide)
                s['reference_inner_m'] = ref
                records.append(s)
                rows.append(dict(route=route, reference_inner_m=ref, band=bands[j], mean=mean[j],
                                 ci_low=ci[0,j], ci_high=ci[1,j], n_units=len(wide), n_blocks=nblocks,
                                 n_targets=int(s.n.sum()), block_km=BLOCK, reps=REPS))
    save(pd.DataFrame(rows), 'distance_common_units_summary.csv')
    save(pd.concat(records, ignore_index=True), 'distance_common_units_detail.csv')


def reference_checks():
    b = read(SRC / 'base_complete_targets.csv.gz')
    p = read(SRC / 'pairs/PFR_annual_targets.csv.gz')
    p = p[p.complete].copy()
    assert b.pixel_id.is_unique and p.pixel_id.is_unique
    cols = [f'D_{year}' for year in range(2020, 2024)]
    b['operation'] = b[cols].mean(axis=1)
    p['operation'] = p[cols].mean(axis=1)
    for x in [b, p]:
        assert x.complete.all() and np.isfinite(x[cols].to_numpy()).all()
    common = b[['pixel_id', 'route', 'unit', 'type', 'operation']].merge(
        p[['pixel_id', 'route', 'unit', 'type', 'operation']], on='pixel_id', suffixes=('_base', '_PFR'), validate='one_to_one')
    for col in ['route','unit','type']:
        assert (common[col+'_base'] == common[col+'_PFR']).all()
        common[col] = common[col+'_base']
    common['PFR_minus_base'] = common.operation_PFR - common.operation_base
    rows = []
    for route, g in common.groupby('route'):
        units = g.groupby('unit')[['operation_base','operation_PFR','PFR_minus_base']].mean()
        mean, ci, nb = unit_boot(units.to_numpy(), units.index.to_numpy())
        for j, metric in enumerate(units.columns):
            rows.append(dict(route=route, metric=metric, mean=mean[j], ci_low=ci[0,j], ci_high=ci[1,j],
                             n_targets=len(g), n_units=len(units), n_blocks=nb))
    save(pd.DataFrame(rows), 'same_targets_local_reference_summary.csv')
    pairs = read(SRC / 'common_strict_PFR_pairs.csv.gz')
    pairs = pairs[pairs.complete].copy()
    original_pairs = len(pairs)
    pairs = pairs[pairs.Russia_target_id.isin(common.pixel_id) & pairs.China_target_id.isin(common.pixel_id)].copy()
    assert pairs.Russia_target_id.is_unique and pairs.China_target_id.is_unique
    indexed = common.set_index('pixel_id')
    r = indexed.loc[pairs.Russia_target_id].reset_index()
    c = indexed.loc[pairs.China_target_id].reset_index()
    assert (r.route == 'Russia').all() and (c.route == 'China').all()
    assert (r.unit.to_numpy() == pairs.Russia_unit.to_numpy()).all()
    assert (c.unit.to_numpy() == pairs.China_unit.to_numpy()).all()
    w = 1 / r.groupby('unit').pixel_id.transform('size').to_numpy()
    w /= w.sum()
    rb, ri = np.unique(r.unit.to_numpy() // BLOCK, return_inverse=True)
    cb, cj = np.unique(c.unit.to_numpy() // BLOCK, return_inverse=True)
    W = np.zeros((len(rb),len(cb)))
    np.add.at(W,(ri,cj),w)
    # Same exact cross-country pairs, weights and bootstrap draws for both references.
    V = np.column_stack([r.operation_base-c.operation_base,
                         r.operation_PFR-c.operation_PFR,
                         r.PFR_minus_base-c.PFR_minus_base])
    S = np.zeros((*W.shape,3)); np.add.at(S,(ri,cj),w[:,None]*V)
    rng = np.random.default_rng(SEED)
    rw = rng.multinomial(len(rb),np.full(len(rb),1/len(rb)),REPS)
    cw = rng.multinomial(len(cb),np.full(len(cb),1/len(cb)),REPS)
    den = np.einsum('bi,ij,bj->b',rw,W,cw)
    num = np.einsum('bi,ijk,bj->bk',rw,S,cw,optimize=True)
    boot = num[den>0]/den[den>0,None]
    ci = np.quantile(boot,[.025,.975],axis=0)
    mean = np.sum(w[:,None]*V,axis=0)
    assert np.isclose(mean[2],mean[1]-mean[0])
    rows = [dict(metric=metric, mean=mean[j], ci_low=ci[0,j], ci_high=ci[1,j],
                 n_pairs=len(pairs), original_PFR_pairs=original_pairs,
                 Russia_units=r.unit.nunique(), China_units=c.unit.nunique(),
                 Russia_blocks=len(rb), China_blocks=len(cb), block_km=BLOCK, reps=REPS)
            for j,metric in enumerate(['country_contrast_base_refs','country_contrast_PFR_refs','reference_change_in_country_contrast'])]
    save(pd.DataFrame(rows),'fixed_country_pairs_reference_comparison.csv')
    pairs['weight']=w
    for j, name in enumerate(['contrast_base_refs','contrast_PFR_refs','PFR_minus_base']):
        pairs[name]=V[:,j]
    save(pairs,'fixed_country_pairs_detail.csv')
    # Add raw components for the exact primary cohort already used in manuscript.
    raw=[]
    for route,g in b.groupby('route'):
        g=g.copy()
        for role in ['near','ref']:
            g[role+'_change'] = g[[f'{role}_{y}' for y in range(2020,2024)]].mean(axis=1)-g[[f'{role}_2013',f'{role}_2014']].mean(axis=1)
        assert np.allclose(g.near_change-g.ref_change,g.operation,atol=1e-7)
        units=g.groupby('unit')[['near_change','ref_change','operation']].mean()
        mean,ci,nb=unit_boot(units.to_numpy(),units.index.to_numpy())
        for j,metric in enumerate(units.columns):
            raw.append(dict(route=route,metric=metric,mean=mean[j],ci_low=ci[0,j],ci_high=ci[1,j],n_targets=len(g),n_units=len(units),n_blocks=nb))
    save(pd.DataFrame(raw),'raw_and_reference_change_verified.csv')


if __name__=='__main__':
    distance_checks()
    reference_checks()
    log={'seed':SEED,'reps':REPS,'block_km':BLOCK,'input_checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
         'interpretation':'Diagnostic of fixed supports. Reference-change check keeps country pairs fixed. It is not a causal decomposition of the full between-specification difference.'}
    (OUT/'analysis_metadata.json').write_text(json.dumps(log,indent=2),encoding='utf8')
    for name in ['distance_common_units_summary.csv','fixed_country_pairs_reference_comparison.csv','same_targets_local_reference_summary.csv']:
        print(name); print(pd.read_csv(OUT/name).to_string(index=False))
