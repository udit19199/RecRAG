import json,collections,re,sys
sys.path.insert(0,'.')
ent=collections.defaultdict(list)
for l in open('runs/retrieval_judge/cache.jsonl'):
    e=json.loads(l)
    if e['dataset']=='hotpotqa': ent[(e['kind'],e['record_id'],e.get('method'))].append(e)
qs={r['id']:r for r in json.load(open('datasets/hotpotqa/hotpotqa_distractor_validation_hard_bridge_1000_seed_42.json'))[:100]}
def yn(a): return a.strip().lower() in('yes','no')
def support(j,yes):
    chain=j['chain']; complete=j['complete'] and bool(chain); ans=j['answer']['status'] if j['answer']['evidence'] else 'missing'
    if complete and (yes or ans=='stated'): return 3
    if ans=='stated': return 2
    if ans=='mentioned' or chain: return 1
    return 0
def sent(st,mains,mentioned):
    if 'contradicted' in st: return 0
    if all(s=='stated' for s in st): return 3
    if any(s=='stated' for s,m in zip(st,mains) if m): return 2
    if mentioned or 'stated' in st: return 1
    return 0
out={}
for m in ['rag','raptor','hipporag','graphrag']:
    for rid,q in qs.items():
        sups=ent[('support',rid,m)]
        s=[support(json.loads(x['response']),yn(q['answer'])) for x in sups]
        cl=json.loads(ent[('claims',rid,None)][-1]['response'])['sentences']
        jd=json.loads(ent[('judge',rid,m)][-1]['response'])['sentences']
        cl={c['sentence']:c for c in cl}
        g=[]
        for js in jd:
            mains=[c['main'] for c in cl[js['sentence']]['claims']]
            st=[c['status'] if (c['status']=='missing' or c['evidence']) else 'missing' for c in sorted(js['claims'],key=lambda c:c['claim'])]
            g.append(sent(st,mains,bool(js['mentions'])))
        out[(m,rid)]=(s,min(g) if g else 0)
json.dump({f'{m}|{r}':v for (m,r),v in out.items()},open('runs/hotpotqa100/scores_per_question.json','w'))
for m in ['rag','raptor','hipporag','graphrag']:
    v=[out[(m,r)] for r in qs]
    print(m,'support first',round(sum(x[0][0] for x in v)/3), 'second',round(sum(x[0][-1] for x in v)/3),'gold',round(sum(x[1] for x in v)/3))
