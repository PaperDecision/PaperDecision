import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# --- 标签标准化函数 ---
def normalize_label(label_str: str) -> str:
    """将各种格式的标签字符串标准化"""
    if not isinstance(label_str, str):
        return "unknown"

    s = label_str.lower().strip()
    if "oral" in s:
        return "oral"
    if "spotlight" in s:
        return "spotlight"
    if "poster" in s:
        return "poster"
    if "reject" in s:
        return "reject"
    if s in ['oral', 'spotlight', 'poster', 'reject', 'accept']:
        return s
    return "unknown"

# --- 改进的 JSON 解析函数 ---
def parse_nested_json(text, field_name="unknown"):
    """尝试解析 Agent 输出中包裹在 Markdown 代码块里的 JSON"""
    if not text:
        return {}
    
    if isinstance(text, dict):
        return text
    
    if not isinstance(text, str):
        return {}
    
    # 尝试多种清理方式
    clean_attempts = [
        text,  # 原始文本
        re.sub(r'```json\s*', '', re.sub(r'```\s*$', '', text)),  # 移除 markdown
        re.sub(r'```json\n?', '', text).replace('```', ''),  # 另一种清理方式
    ]
    
    for attempt in clean_attempts:
        try:
            result = json.loads(attempt.strip())
            return result
        except json.JSONDecodeError:
            continue
    
    return {}

def load_data(file_paths):
    data = []
    parse_stats = {
        'total': 0,
        'visual_content_success': 0,
        'review_analysis_success': 0,
        'rebuttal_analysis_success': 0,
        'decision_success': 0
    }
    
    # 新增：全局审稿人统计
    global_reviewer_stats = {
        'total_experts': 0,
        'total_non_experts': 0,
        'total_reviewers': 0
    }
    
    for path in file_paths:
        model_name = "analysis_result"
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行JSON格式错误，已跳过。")
                    continue
                
                parse_stats['total'] += 1
                
                # 提取嵌套的 Agent 分析数据
                final_input = item.get('final_input', {})
                
                # 解析各个嵌套字段
                paper_analysis = parse_nested_json(
                    final_input.get('visual_content_eval', '{}'), 
                    'visual_content_eval'
                )
                if paper_analysis:
                    parse_stats['visual_content_success'] += 1
                
                review_analysis = parse_nested_json(
                    final_input.get('review_analysis', '{}'),
                    'review_analysis'
                )
                if review_analysis:
                    parse_stats['review_analysis_success'] += 1
                
                rebuttal_analysis = parse_nested_json(
                    final_input.get('rebuttal_analysis', '{}'),
                    'rebuttal_analysis'
                )
                if rebuttal_analysis:
                    parse_stats['rebuttal_analysis_success'] += 1
                
                # decision 字段已经是 dict，不需要解析
                ac_decision = item.get('decision', {})
                if isinstance(ac_decision, str):
                    ac_decision = parse_nested_json(ac_decision, 'decision')
                if ac_decision:
                    parse_stats['decision_success'] += 1
                
                # 标准化 ground truth
                raw_gt = item.get('ground_truth', 'unknown')
                normalized_gt = normalize_label(raw_gt)

                # 提取 reviewer scores（从 final_input.raw_scores）
                raw_scores = final_input.get('raw_scores', [])
                avg_score = sum(raw_scores) / len(raw_scores) if raw_scores else None
                
                # 展平数据
                entry = {
                    'model': model_name,
                    'title': item.get('title', 'No Title'),
                    'ground_truth': normalized_gt,
                    'pred_decision': normalize_label(ac_decision.get('final_decision', 'Reject')),
                    'pred_score': ac_decision.get('final_score'),
                    
                    # Decision 相关
                    'decision_archetype': ac_decision.get('decision_archetype', 'Unknown'),
                    'key_factor': ac_decision.get('key_factor', 'Unknown'),
                    'confidence': ac_decision.get('confidence', 'Unknown'),
                    
                    # Paper Agent Features
                    'novelty': paper_analysis.get('novelty_level', 'Unknown'),
                    'award_potential': paper_analysis.get('award_potential', 'Unknown'),
                    'visual_quality': paper_analysis.get('visual_quality', 'Unknown'),
                    
                    # Review Agent Features
                    'consensus': review_analysis.get('consensus_type', 'Unknown'),
                    'risk_level': review_analysis.get('risk_level', 'Unknown'),
                    'fatal_flaw': review_analysis.get('most_dangerous_issue', 'None'),
                    
                    # Rebuttal Agent Features
                    'rebuttal_strength': rebuttal_analysis.get('rebuttal_effectiveness', 'Unknown'),
                    'success_rate': rebuttal_analysis.get('success_rate', 0.0),
                    'admitted_fatal_error': rebuttal_analysis.get('admitted_fatal_error', False),
                    'author_self_sabotage': rebuttal_analysis.get('author_self_sabotage', False),
                    
                    # Reviewer Scores
                    'raw_scores': raw_scores,
                    'avg_reviewer_score': avg_score,
                    'min_score': min(raw_scores) if raw_scores else None,
                    'max_score': max(raw_scores) if raw_scores else None,
                    'score_spread': (max(raw_scores) - min(raw_scores)) if raw_scores else None,
                    
                    # 统计值
                    'converted_reviewers': 0,
                    'expert_score': None,
                    'num_experts': 0,
                    'num_non_experts': 0,  # 新增：每篇论文的非expert数量
                    'num_fatal_flaws': 0,
                }
                
                # 统计 Expert 分数
                # 1. 收集所有审稿人的分数和身份
                reviewers = review_analysis.get('reviewer_analysis', {})
                expert_score_value = None
                non_expert_scores = []
                paper_expert_count = 0
                paper_non_expert_count = 0

                if isinstance(reviewers, dict):
                    for r_name, r_data in reviewers.items():
                        if isinstance(r_data, list) and len(r_data) > 0:
                            r_data = r_data[0]
                        elif not isinstance(r_data, dict):
                            continue
                        
                        score = r_data.get('score')
                        
                        # 2. 判断是否是专家，并分别存储分数
                        if r_data.get('credibility') == 'Expert':
                            paper_expert_count += 1
                            global_reviewer_stats['total_experts'] += 1
                            if score is not None:
                                expert_score_value = score
                        else:
                            paper_non_expert_count += 1
                            global_reviewer_stats['total_non_experts'] += 1
                            if score is not None:
                                non_expert_scores.append(score)
                        
                        global_reviewer_stats['total_reviewers'] += 1
                
                # 3. 计算并赋值
                entry['expert_score'] = expert_score_value
                entry['num_experts'] = paper_expert_count
                entry['num_non_experts'] = paper_non_expert_count
                
                # 只有在有非专家分数的情况下才计算平均值
                if non_expert_scores:
                    entry['avg_reviewer_score'] = sum(non_expert_scores) / len(non_expert_scores)
                
                # 统计 Fatal Flaws 数量
                fatal_flaws = review_analysis.get('fatal_flaw_allegations', [])
                if isinstance(fatal_flaws, list):
                    entry['num_fatal_flaws'] = len(fatal_flaws)
                
                # 统计 Rebuttal 转化情况
                final_states = rebuttal_analysis.get('reviewer_final_states', {})
                if isinstance(final_states, dict):
                    for r_name, r_state in final_states.items():
                        if isinstance(r_state, dict):
                            state = r_state.get('final_state', '')
                            # 各种可能的正面转化状态
                            if state in ['Converted_From_Low', 'Softened', 'Converted']:
                                entry['converted_reviewers'] += 1
                        
                data.append(entry)
    
    # 打印解析统计
    print("\n" + "=" * 50)
    print("数据解析统计:")
    print("=" * 50)
    print(f"总记录数: {parse_stats['total']}")
    print(f"visual_content_eval 解析成功: {parse_stats['visual_content_success']} ({parse_stats['visual_content_success']/parse_stats['total']*100:.1f}%)")
    print(f"review_analysis 解析成功: {parse_stats['review_analysis_success']} ({parse_stats['review_analysis_success']/parse_stats['total']*100:.1f}%)")
    print(f"rebuttal_analysis 解析成功: {parse_stats['rebuttal_analysis_success']} ({parse_stats['rebuttal_analysis_success']/parse_stats['total']*100:.1f}%)")
    print(f"decision 解析成功: {parse_stats['decision_success']} ({parse_stats['decision_success']/parse_stats['total']*100:.1f}%)")
    print("=" * 50)
    
    # 新增：打印审稿人统计
    print("\n" + "=" * 50)
    print("审稿人统计:")
    print("=" * 50)
    print(f"Expert 审稿人总数: {global_reviewer_stats['total_experts']}")
    print(f"Non-Expert 审稿人总数: {global_reviewer_stats['total_non_experts']}")
    print(f"所有审稿人总数: {global_reviewer_stats['total_reviewers']}")
    if global_reviewer_stats['total_reviewers'] > 0:
        expert_ratio = global_reviewer_stats['total_experts'] / global_reviewer_stats['total_reviewers'] * 100
        print(f"Expert 占比: {expert_ratio:.1f}%")
    print("=" * 50 + "\n")
    
    return pd.DataFrame(data), global_reviewer_stats  # 返回统计信息

# ==========================================
# 主程序
# ==========================================

files = ['iclr2025_all_gt.jsonl']
df, reviewer_stats = load_data(files)  # 接收统计信息

# 将 Ground Truth 映射为 0/1
df['is_accepted'] = df['ground_truth'].apply(lambda x: 1 if x in ['oral', 'spotlight', 'poster', 'accept'] else 0)

print(f"\nTotal Papers Analyzed: {len(df)}")
print(f"\nGround Truth Distribution:")
print(df['ground_truth'].value_counts())
print(f"\nPrediction Distribution:")
print(df['pred_decision'].value_counts())
print("-" * 50)

# ==========================================
# Analysis 1: Rebuttal 有效性分析
# ==========================================
print("\n[Analysis 1] Rebuttal Effectiveness vs. Acceptance Rate")

# 过滤掉 Unknown
df_rebuttal = df[df['rebuttal_strength'] != 'Unknown']
if not df_rebuttal.empty:
    pivot_rebuttal = df_rebuttal.groupby('rebuttal_strength')['is_accepted'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print(pivot_rebuttal)
else:
    print("No valid rebuttal_strength data found.")

# 转化审稿人的影响
print("\nImpact of Converting at least 1 Reviewer:")
conversion_impact = df.groupby(df['converted_reviewers'] > 0)['is_accepted'].agg(['mean', 'count'])
conversion_impact.index = ['No Conversion', 'At Least 1 Converted']
print(conversion_impact)

# ==========================================
# Analysis 2: 专家分数分析
# ==========================================
df_expert = df[df['expert_score'].notna()].copy()

if not df_expert.empty:
    print("\n[Analysis 2] Expert Score Correlation")
    # 确保分数列是数值类型
    df_expert['expert_score'] = pd.to_numeric(df_expert['expert_score'], errors='coerce')
    df_expert['avg_reviewer_score'] = pd.to_numeric(df_expert['avg_reviewer_score'], errors='coerce')
    df_expert.dropna(subset=['expert_score', 'avg_reviewer_score'], inplace=True)

    corr_expert = df_expert['expert_score'].corr(df_expert['is_accepted'])
    corr_avg = df_expert['avg_reviewer_score'].corr(df_expert['is_accepted'])
    
    print(f"Papers with identified experts: {len(df_expert)}")
    print(f"Correlation (Expert Score vs Accept): {corr_expert:.4f}")
    print(f"Correlation (Non-Expert Avg Score vs Accept): {corr_avg:.4f}")
    
    if corr_expert > corr_avg:
        print(">> Insight: Expert opinion predicts acceptance better than the non-expert average score.")
    else:
        print(">> Insight: The crowd wisdom (non-expert average) matters more than the single expert.")

    # ==========================================
    # [新增] Insight 2.1: 极端专家意见的影响
    # ==========================================
    print("\n[Analysis 2.1] Impact of Extreme Expert Opinions")

    HIGH_SCORE_THRESHOLD = 7
    LOW_SCORE_THRESHOLD = 4

    # 1. 专家给出高分 (>= 7) 的情况
    high_expert_score_df = df_expert[df_expert['expert_score'] >= HIGH_SCORE_THRESHOLD]
    high_expert_acceptance_rate = high_expert_score_df['is_accepted'].mean()
    high_expert_count = len(high_expert_score_df)

    print(f"\nWhen an Expert gives a high score (>= {HIGH_SCORE_THRESHOLD}):")
    if high_expert_count > 0:
        print(f"  - Acceptance Rate: {high_expert_acceptance_rate:.2%}")
        print(f"  - Number of papers: {high_expert_count}")
    else:
        print("  - No papers found in this category.")

    # 2. 专家给出低分 (<= 4) 的情况
    low_expert_score_df = df_expert[df_expert['expert_score'] <= LOW_SCORE_THRESHOLD]
    low_expert_acceptance_rate = low_expert_score_df['is_accepted'].mean()
    low_expert_count = len(low_expert_score_df)
    
    print(f"\nWhen an Expert gives a low score (<= {LOW_SCORE_THRESHOLD}):")
    if low_expert_count > 0:
        print(f"  - Acceptance Rate: {low_expert_acceptance_rate:.2%}")
        print(f"  - Number of papers: {low_expert_count}")
    else:
        print("  - No papers found in this category.")

    # 3. 作为对比，查看所有论文的平均接受率
    overall_acceptance_rate = df['is_accepted'].mean()
    print(f"\nFor reference, the overall acceptance rate for all papers is: {overall_acceptance_rate:.2%}")

    # 4. [高级分析] 专家一票否决权？
    avg_high_score = 6
    controversial_papers = low_expert_score_df[low_expert_score_df['avg_reviewer_score'] > avg_high_score]
    controversial_acceptance_rate = controversial_papers['is_accepted'].mean()
    controversial_count = len(controversial_papers)

    print(f"\n[Deeper Dive] What happens when Expert gives a low score (<= {LOW_SCORE_THRESHOLD}) but non-expert average is high (> {avg_high_score})?")
    if controversial_count > 0:
        print(f"  - Acceptance Rate: {controversial_acceptance_rate:.2%}")
        print(f"  - Number of such controversial papers: {controversial_count}")
        print("  >> Insight: This shows if an expert's low score acts as a 'veto' even when other reviewers are positive.")
    else:
        print("  - No such controversial papers found.")

# ==========================================
# Analysis 2.5: 每篇论文的 Expert 分布
# ==========================================
print("\n[Analysis 2.5] Expert Distribution per Paper")
print(f"\nNumber of Experts per Paper:")
print(df['num_experts'].value_counts().sort_index())
print(f"\nNumber of Non-Experts per Paper:")
print(df['num_non_experts'].value_counts().sort_index())

# 有 expert 的论文 vs 无 expert 的论文
df['has_expert'] = df['num_experts'] > 0
has_expert_stats = df.groupby('has_expert')['is_accepted'].agg(['mean', 'count'])
has_expert_stats.index = ['No Expert', 'Has Expert']
print(f"\nAcceptance Rate by Expert Presence:")
print(has_expert_stats)

# ==========================================
# Analysis 3: 致命伤致死率
# ==========================================
print("\n[Analysis 3] Most Dangerous Issues (Risk of Rejection)")

df_flaw = df[df['fatal_flaw'].notna() & (df['fatal_flaw'] != 'None') & (df['fatal_flaw'] != 'Unknown')]
if not df_flaw.empty:
    flaw_stats = df_flaw.groupby('fatal_flaw')['is_accepted'].agg(['mean', 'count'])
    flaw_stats = flaw_stats[flaw_stats['count'] >= 3]
    flaw_stats = flaw_stats.sort_values('mean')
    print("Top Fatal Flaws (by lowest acceptance rate, min 3 occurrences):")
    print(flaw_stats.head(10))
else:
    print("No fatal flaw data found.")

# ==========================================
# Analysis 4: Oral 论文特征
# ==========================================
print("\n[Analysis 4] Characteristics of 'Oral' Papers")

oral_papers = df[df['ground_truth'] == 'oral']
if not oral_papers.empty:
    print(f"Total Oral papers: {len(oral_papers)}")
    print(f"\nNovelty Distribution:")
    print(oral_papers['novelty'].value_counts())
    print(f"\nAward Potential Distribution:")
    print(oral_papers['award_potential'].value_counts())
    print(f"\nAverage Reviewer Score: {oral_papers['avg_reviewer_score'].mean():.2f}")
else:
    print("No 'oral' papers found in the dataset.")

# ==========================================
# Analysis 5: Decision Archetype 分析
# ==========================================
print("\n[Analysis 5] Decision Archetype Analysis")

df_archetype = df[df['decision_archetype'] != 'Unknown']
if not df_archetype.empty:
    archetype_stats = df_archetype.groupby('decision_archetype').agg({
        'is_accepted': ['mean', 'count'],
        'avg_reviewer_score': 'mean'
    }).round(3)
    archetype_stats.columns = ['acceptance_rate', 'count', 'avg_score']
    archetype_stats = archetype_stats.sort_values('count', ascending=False)
    print(archetype_stats.head(10))
else:
    print("No decision archetype data found.")

# ==========================================
# Analysis 6: 预测准确率
# ==========================================
print("\n[Analysis 6] Prediction Accuracy")

df['gt_binary'] = df['ground_truth'].apply(lambda x: 'accept' if x in ['oral', 'spotlight', 'poster'] else 'reject')
df['pred_binary'] = df['pred_decision'].apply(lambda x: 'accept' if x in ['oral', 'spotlight', 'poster'] else 'reject')

accuracy = (df['gt_binary'] == df['pred_binary']).mean()
print(f"Binary Accuracy (Accept/Reject): {accuracy:.2%}")

print("\nConfusion Matrix:")
confusion = pd.crosstab(df['gt_binary'], df['pred_binary'], margins=True)
print(confusion)

# ==========================================
# Analysis 7: 分数分布与接受率
# ==========================================
print("\n[Analysis 7] Score Distribution vs Acceptance")

df_scores = df[df['avg_reviewer_score'].notna()].copy()
if not df_scores.empty:
    df_scores['score_bin'] = pd.cut(df_scores['avg_reviewer_score'], 
                                     bins=[0, 4, 5, 6, 7, 10],
                                     labels=['<4', '4-5', '5-6', '6-7', '>7'])
    
    score_analysis = df_scores.groupby('score_bin')['is_accepted'].agg(['mean', 'count'])
    print(score_analysis)