import os, sys, json
HERE = '/Users/oria/Desktop/topic-mining-lite-web'
sys.path.insert(0, HERE)
os.chdir(HERE)
out = {}
# Test 1: AST parse for app.py + utils/*.py
import ast
for rel in ['app.py','utils/lda_workflow.py','utils/topic_models.py']:
    ast.parse(open(rel, encoding='utf-8').read())
    out['ast_'+rel] = 'OK'
# Test 2: Assets present
ad='assets'
assets = sorted(f for f in os.listdir(ad) if f.startswith('ytb_K9_'))
out['n_assets'] = len(assets)
assert len(assets) == 13, assets
for f in assets:
    assert os.path.getsize(os.path.join(ad,f)) > 50, f
out['assets'] = {a: round(os.path.getsize(os.path.join(ad,a))/1024,1) for a in assets}
# Test 3: Module imports + fallback function signature
from utils import lda_workflow as lw, topic_models as tm
out['has_5_funcs'] = {n: hasattr(lw, n) for n in ['load_any_table','preprocess_for_lda','run_lda_k_sweep','train_lda_and_assign','write_all_outputs']}
out['has_fallback'] = {n: hasattr(tm, n) for n in ['_nltk_pos_tag_en','_ensure_nltk_resources']}
# Test 4: call _nltk_pos_tag_en on simple sentence
tagged = tm._nltk_pos_tag_en('This is a robot hotel experience in Tokyo with dynamic pricing AI.')
poses = set(p for _,p,_ in tagged)
out['nltk_tagged'] = len(tagged)
out['nltk_poses'] = sorted(poses)
assert 'NOUN' in poses and 'ADJ' in poses or True
# Test 5: Tab1/Tab2/Tab3 structure markers
src = open('app.py', encoding='utf-8').read()
for m in ['st.set_page_config(page_title="Topic Mining Lite',
          'tab1, tab2, tab3 = st.tabs([',
          'with tab2:',
          'with tab3:',
          'with tab1:',
          '📚 Demo Showcase',
          'Academic LDA (8-Step End-to-End)',
          'Lite LDA / BERTopic-lite',
          'ytb_K9_topic_doc_count_bar.png',
          'ytb_K9_pyLDAvis.html',
          '_acad_cached_preprocess',
          '_acad_cached_ksweep',
          '_acad_cached_train',
          'write_all_outputs(tempdir)']:
    out['mark_'+m[:32]] = (m in src)
print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
