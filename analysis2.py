import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import re


def normalize_ground_truth(gt: str) -> tuple[str, int]:
    """返回 (normalized_label, is_accepted)"""
    gt = gt.strip()
    if gt == "Reject":
        return "reject", 0
    elif gt == "Accept (Oral)":
        return "oral", 1
    elif gt == "Accept (Spotlight)":
        return "spotlight", 1
    elif gt == "Accept (Poster)":
        return "poster", 1
    else:
        return "unknown", 0
    
def parse_nested_json(text):
    """尝试解析 Agent 输出中包裹在 Markdown 代码块里的 JSON"""
    if not isinstance(text, str):
        return {}
    
    if not text.strip():
        return {}
    
    # 移除可能的 markdown 代码块标记（多种格式）
    clean_text = text.strip()
    
    # 处理 "json\n{" 开头的情况（没有 ```）
    if clean_text.startswith('json\n'):
        clean_text = clean_text[5:]  # 移除 "json\n"
    elif clean_text.startswith('json{'):
        clean_text = clean_text[4:]  # 移除 "json"
    
    # 处理标准的 ```json ... ``` 格式
    clean_text = re.sub(r'^```json\s*\n?', '', clean_text)
    clean_text = re.sub(r'^```\s*\n?', '', clean_text)
    clean_text = re.sub(r'\n?```\s*$', '', clean_text)
    
    # 尝试找到 JSON 对象的开始和结束
    clean_text = clean_text.strip()
    
    # 如果文本不是以 { 开头，尝试找到第一个 {
    if not clean_text.startswith('{'):
        start_idx = clean_text.find('{')
        if start_idx != -1:
            clean_text = clean_text[start_idx:]
    
    # 尝试解析
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        # 尝试修复常见问题
        try:
            # 有时候末尾有多余字符
            # 找到最后一个 } 的位置
            last_brace = clean_text.rfind('}')
            if last_brace != -1:
                clean_text = clean_text[:last_brace + 1]
                return json.loads(clean_text)
        except:
            pass
        
        # 调试输出
        # print(f"JSON Parse Error: {e}")
        # print(f"First 200 chars: {clean_text[:200]}")
        return {}

def safe_get(d, *keys, default=None):
    """安全地从嵌套字典中获取值"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default

def load_data(file_paths):
    data = []
    reviewer_state_records = []
    parse_errors = []
    
    for path in file_paths:
        model_name = path.split('/')[-1].replace('.jsonl', '')
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    parse_errors.append(f"{path}:{line_num} - Line parse error: {e}")
                    continue
                
                final_input = item.get('final_input', {})
                
                # 解析嵌套 JSON
                paper_analysis = parse_nested_json(final_input.get('visual_content_eval', '{}'))
                review_analysis = parse_nested_json(final_input.get('review_analysis', '{}'))
                rebuttal_analysis = parse_nested_json(final_input.get('rebuttal_analysis', '{}'))
                ac_decision = item.get('decision', {})
                
                # 调试：检查解析结果
                if not paper_analysis and final_input.get('visual_content_eval'):
                    parse_errors.append(f"{path}:{line_num} - Failed to parse visual_content_eval")
                if not review_analysis and final_input.get('review_analysis'):
                    parse_errors.append(f"{path}:{line_num} - Failed to parse review_analysis")
                if not rebuttal_analysis and final_input.get('rebuttal_analysis'):
                    parse_errors.append(f"{path}:{line_num} - Failed to parse rebuttal_analysis")
                
                # 获取审稿人信息
                reviewers = safe_get(review_analysis, 'reviewer_analysis', default={})
                final_states = safe_get(rebuttal_analysis, 'reviewer_final_states', default={})
                
                # 从 raw_scores 获取分数（如果 reviewer_analysis 解析失败）
                raw_scores = final_input.get('raw_scores', [])
                
                scores = []
                has_expert = False
                expert_score = None
                credibility_distribution = Counter()
                reviewer_types = Counter()
                
                # 如果有 reviewer_analysis，从中提取信息
                if reviewers:
                    for r_name, r_data in reviewers.items():
                        if isinstance(r_data, dict):
                            score = r_data.get('score', 0)
                            if score:
                                scores.append(score)
                            credibility = r_data.get('credibility', 'Unknown')
                            reviewer_type = r_data.get('type', 'Unknown')
                            
                            credibility_distribution[credibility] += 1
                            reviewer_types[reviewer_type] += 1
                            
                            if credibility == 'Expert':
                                has_expert = True
                                expert_score = score
                
                # 如果没有从 reviewer_analysis 获取到分数，使用 raw_scores
                if not scores and raw_scores:
                    scores = [s for s in raw_scores if isinstance(s, (int, float))]
                
                # 计算分数统计
                if scores:
                    avg_reviewer_score = sum(scores) / len(scores)
                    score_variance = np.var(scores) if len(scores) > 1 else 0
                    score_range = max(scores) - min(scores)
                    min_score = min(scores)
                    max_score = max(scores)
                else:
                    avg_reviewer_score = 0
                    score_variance = 0
                    score_range = 0
                    min_score = 0
                    max_score = 0
                
                # 处理 reviewer states
                state_counts = Counter()
                converted_reviewers = 0
                
                if final_states:
                    for r_name, r_state in final_states.items():
                        if isinstance(r_state, dict):
                            final_state = r_state.get('final_state', 'Unknown')
                            initial_score = r_state.get('initial_score', 0)
                            
                            state_counts[final_state] += 1
                            
                            if final_state == 'Converted_From_Low':
                                converted_reviewers += 1
                            
                            raw_gt = item.get('ground_truth', '')
                            norm_gt, is_accepted_val = normalize_ground_truth(raw_gt)

                            reviewer_state_records.append({
                                'model': model_name,
                                'title': item.get('title', ''),
                                'reviewer': r_name,
                                'final_state': final_state,
                                'initial_score': initial_score,
                                'ground_truth': norm_gt.lower(),
                                'is_accepted': 1 if norm_gt.lower() in ['oral', 'spotlight', 'poster', 'accept'] else 0
                            })
                
                # 处理 consensus_type
                consensus_type = safe_get(review_analysis, 'consensus_type', default='Unknown')
                if consensus_type == 'Strong Consensus':
                    if avg_reviewer_score >= 6:
                        consensus_type = 'Strong Consensus (score >= 6)'
                    else:
                        consensus_type = 'Strong Consensus (score < 6)'
                
                # 处理 fatal_flaw_allegations
                fatal_flaws = safe_get(review_analysis, 'fatal_flaw_allegations', default=[])
                fatal_flaw_types = []
                if isinstance(fatal_flaws, list):
                    for flaw in fatal_flaws:
                        if isinstance(flaw, dict):
                            flaw_type = flaw.get('type', '')
                            if flaw_type:
                                fatal_flaw_types.append(flaw_type)
                
                raw_gt = item.get('ground_truth', '')
                norm_gt, is_accepted_val = normalize_ground_truth(raw_gt)
                entry = {
                    'model': model_name,
                    'title': item.get('title', ''),
                    'ground_truth': norm_gt.lower(),
                    'pred_decision': safe_get(ac_decision, 'final_decision', default='Reject').lower(),
                    'pred_score': float(ac_decision.get('final_score')) if ac_decision.get('final_score') else None,
                    'decision_archetype': safe_get(ac_decision, 'decision_archetype', default='Unknown'),
                    'key_factor': safe_get(ac_decision, 'key_factor', default='Unknown'),
                    
                    # Paper Agent Features
                    'novelty': safe_get(paper_analysis, 'novelty_level', default='Unknown'),
                    'award_potential': safe_get(paper_analysis, 'award_potential', default='Unknown'),
                    'visual_quality': safe_get(paper_analysis, 'visual_quality', default='Unknown'),
                    
                    # Review Agent Features
                    'consensus': consensus_type,
                    'risk_level': safe_get(review_analysis, 'risk_level', default='Unknown'),
                    'fatal_flaw': safe_get(review_analysis, 'most_dangerous_issue', default='None'),
                    'fatal_flaw_types': fatal_flaw_types,
                    
                    # Reviewer Composition
                    'num_experts': credibility_distribution.get('Expert', 0),
                    'num_competent': credibility_distribution.get('Competent', 0),
                    'num_shallow': credibility_distribution.get('Shallow', 0),
                    'num_empty_praise': reviewer_types.get('Empty_Praise', 0),
                    'num_high_credibility_low_score': reviewer_types.get('High_Credibility_Low_Score', 0),
                    
                    # Rebuttal Agent Features
                    'rebuttal_strength': safe_get(rebuttal_analysis, 'rebuttal_effectiveness', default='None'),
                    'success_rate': safe_get(rebuttal_analysis, 'success_rate', default=0.0),
                    'admitted_fatal_error': safe_get(rebuttal_analysis, 'admitted_fatal_error', default=False),
                    'author_self_sabotage': safe_get(rebuttal_analysis, 'author_self_sabotage', default=False),
                    'converted_reviewers': converted_reviewers,
                    
                    # Reviewer Score Statistics
                    'has_expert': has_expert,
                    'expert_score': expert_score,
                    'avg_reviewer_score': avg_reviewer_score,
                    'min_reviewer_score': min_score,
                    'max_reviewer_score': max_score,
                    'score_variance': score_variance,
                    'score_range': score_range,
                    
                    # Reviewer States
                    'count_softened': state_counts.get('Softened', 0),
                    'count_explicitly_stubborn': state_counts.get('Explicitly_Stubborn', 0),
                    'count_ghosted_but_overruled': state_counts.get('Ghosted_But_Overruled', 0),
                    'count_ghosted_still_dangerous': state_counts.get('Ghosted_Still_Dangerous', 0),
                    'count_converted_from_low': state_counts.get('Converted_From_Low', 0),
                    'count_new_red_flag_raised': state_counts.get('New_Red_Flag_Raised', 0),
                    
                    'total_reviewers': max(len(final_states), len(reviewers), len(raw_scores))
                }
                        
                data.append(entry)
    
    # 打印解析错误摘要
    if parse_errors:
        print(f"\n⚠️  Parse Warnings ({len(parse_errors)} total):")
        for err in parse_errors[:10]:  # 只显示前10个
            print(f"  - {err}")
        if len(parse_errors) > 10:
            print(f"  ... and {len(parse_errors) - 10} more")
    
    return pd.DataFrame(data), pd.DataFrame(reviewer_state_records)

# ==========================================
# 核心分析函数
# ==========================================

def print_section_header(title):
    """打印格式化的分析标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def analyze_acceptance_by_category(df, category_col, title, top_n=None):
    """通用的分类变量接受率分析"""
    print(f"\n{title}")
    print("-" * 80)
    
    # 过滤掉 Unknown 和空值
    valid_df = df[df[category_col].notna() & (df[category_col] != 'Unknown') & (df[category_col] != '')]
    
    if len(valid_df) == 0:
        print("  No valid data found for this category")
        return None
    
    result = valid_df.groupby(category_col).agg({
        'is_accepted': ['mean', 'count'],
        'ground_truth': lambda x: (x == 'oral').sum()
    }).round(3)
    
    result.columns = ['Accept_Rate', 'Count', 'Num_Oral']
    result = result.sort_values('Accept_Rate', ascending=False)
    
    if top_n:
        result = result.head(top_n)
    
    print(result)
    
    # 打印 Unknown 的数量
    unknown_count = len(df) - len(valid_df)
    if unknown_count > 0:
        print(f"\n  (Note: {unknown_count} papers with Unknown/empty values excluded)")
    
    return result

def analyze_score_distribution(df, title):
    """分数分布分析"""
    print(f"\n{title}")
    print("-" * 80)
    
    for decision in ['oral', 'spotlight', 'poster', 'reject']:
        subset = df[df['ground_truth'] == decision]
        if len(subset) > 0:
            valid_subset = subset[subset['avg_reviewer_score'] > 0]
            if len(valid_subset) > 0:
                print(f"{decision.upper():12s} (n={len(subset):3d}): "
                      f"Avg={valid_subset['avg_reviewer_score'].mean():.2f}, "
                      f"Min={valid_subset['min_reviewer_score'].mean():.2f}, "
                      f"Max={valid_subset['max_reviewer_score'].mean():.2f}, "
                      f"Range={valid_subset['score_range'].mean():.2f}")
            else:
                print(f"{decision.upper():12s} (n={len(subset):3d}): No valid score data")

def analyze_correlation_matrix(df, features, title):
    """相关性矩阵分析"""
    print(f"\n{title}")
    print("-" * 80)
    
    # 只选择数值型且非全空的列
    valid_features = []
    for f in features:
        if f in df.columns and df[f].notna().sum() > 0:
            valid_features.append(f)
    
    if 'is_accepted' not in valid_features:
        print("  Cannot compute correlation: 'is_accepted' column missing or empty")
        return None
    
    numeric_df = df[valid_features].apply(pd.to_numeric, errors='coerce')
    
    if numeric_df.empty:
        print("  No valid numeric data for correlation")
        return None
    
    corr_matrix = numeric_df.corr()['is_accepted'].sort_values(ascending=False)
    print(corr_matrix)
    return corr_matrix

def print_data_quality_report(df):
    """打印数据质量报告"""
    print_section_header("DATA QUALITY REPORT")
    
    print("\n[Data Completeness Check]")
    print("-" * 80)
    
    key_fields = [
        ('novelty', 'Paper Analysis'),
        ('award_potential', 'Paper Analysis'),
        ('consensus', 'Review Analysis'),
        ('risk_level', 'Review Analysis'),
        ('fatal_flaw', 'Review Analysis'),
        ('rebuttal_strength', 'Rebuttal Analysis'),
        ('decision_archetype', 'AC Decision'),
    ]
    
    for field, source in key_fields:
        if field in df.columns:
            valid = df[field].notna() & (df[field] != 'Unknown') & (df[field] != '') & (df[field] != 'None')
            pct = valid.sum() / len(df) * 100
            print(f"  {field:30s} ({source:20s}): {valid.sum():4d}/{len(df)} ({pct:5.1f}%)")
        else:
            print(f"  {field:30s}: MISSING COLUMN")
    
    print("\n[Score Data Check]")
    print("-" * 80)
    has_scores = df['avg_reviewer_score'] > 0
    print(f"  Papers with valid scores: {has_scores.sum()}/{len(df)} ({has_scores.mean()*100:.1f}%)")
    
    if has_scores.sum() > 0:
        print(f"  Score range: {df.loc[has_scores, 'min_reviewer_score'].min():.1f} - {df.loc[has_scores, 'max_reviewer_score'].max():.1f}")
        print(f"  Average score: {df.loc[has_scores, 'avg_reviewer_score'].mean():.2f}")

# ==========================================
# 主分析脚本
# ==========================================

files = [
    'iclr2025_all_gt.jsonl'
]

try:
    df, df_reviewer_states = load_data(files)
    df['is_accepted'] = df['ground_truth'].apply(lambda x: 1 if x in ['oral', 'spotlight', 'poster', 'accept'] else 0)
    
    print("\n" + "=" * 80)
    print("  ICLR 2025 Paper Analysis - Data Loading Complete")
    print("=" * 80)
    
    print(f"\nTotal Papers: {len(df)}")
    print(f"Accept Rate: {df['is_accepted'].mean():.1%}")
    print(f"\nGround Truth Distribution:")
    print(df['ground_truth'].value_counts())
    
    # 先打印数据质量报告
    print_data_quality_report(df)

    # ==========================================
    # SECTION 1: 论文质量信号分析
    # ==========================================
    print_section_header("SECTION 1: Paper Quality Signals")
    
    # 1.1 Novelty Level 的决定性作用
    analyze_acceptance_by_category(df, 'novelty', 
        "[1.1] Novelty Level vs Acceptance (The Holy Grail)")
    
    # 1.2 Award Potential 的真实价值
    analyze_acceptance_by_category(df, 'award_potential',
        "[1.2] Award Potential vs Acceptance")
    
    # 1.3 Novelty + Award Potential 交叉分析
    print("\n[1.3] Novelty × Award Potential Cross-Analysis")
    print("-" * 80)
    
    # 过滤有效数据
    valid_for_cross = df[
        (df['novelty'].notna()) & (df['novelty'] != 'Unknown') &
        (df['award_potential'].notna()) & (df['award_potential'] != 'Unknown')
    ]
    
    if len(valid_for_cross) > 0:
        cross_tab = pd.crosstab(
            valid_for_cross['novelty'], 
            valid_for_cross['award_potential'], 
            values=valid_for_cross['is_accepted'], 
            aggfunc='mean'
        ).round(3)
        print(cross_tab)
    else:
        print("  Insufficient data for cross-analysis")
    
    # 1.4 Oral 论文的共同特征
    print("\n[1.4] What Makes an ORAL Paper? (Multi-dimensional Profile)")
    print("-" * 80)
    oral_papers = df[(df['ground_truth'] == 'oral') | (df['ground_truth'] == 'spotlight')]
    if len(oral_papers) > 0:
        print(f"Total Oral/Spotlight Papers: {len(oral_papers)}\n")
        
        # 只显示有效的分布
        for col, name in [('novelty', 'Novelty'), ('award_potential', 'Award Potential'), ('consensus', 'Consensus')]:
            valid = oral_papers[oral_papers[col].notna() & (oral_papers[col] != 'Unknown')]
            if len(valid) > 0:
                print(f"{name} Distribution:")
                print(valid[col].value_counts(normalize=True).round(3))
                print()
        
        valid_scores = oral_papers[oral_papers['avg_reviewer_score'] > 0]
        if len(valid_scores) > 0:
            print(f"Average Reviewer Score: {valid_scores['avg_reviewer_score'].mean():.2f}")
            print(f"Score Variance: {valid_scores['score_variance'].mean():.2f}")
        
        if oral_papers['num_experts'].sum() > 0:
            print(f"% with Expert Reviewer: {(oral_papers['num_experts'] > 0).mean():.1%}")
    else:
        print("  No Oral/Spotlight papers found")

    # ==========================================
    # SECTION 2: 审稿人行为模式分析
    # ==========================================
    print_section_header("SECTION 2: Reviewer Behavior Patterns")
    
    analyze_acceptance_by_category(df, 'consensus',
        "[2.1] Consensus Type vs Acceptance")
    
    analyze_acceptance_by_category(df, 'risk_level',
        "[2.2] Risk Level Assessment")
    
    analyze_acceptance_by_category(df, 'fatal_flaw',
        "[2.3] Fatal Flaw Types (Ordered by Lethality)", top_n=10)
    
    # 2.4 审稿人构成的影响
    print("\n[2.4] Reviewer Panel Composition Impact")
    print("-" * 80)
    
    if df['num_experts'].sum() > 0:
        expert_impact = df.groupby(df['num_experts'] > 0)['is_accepted'].agg(['mean', 'count'])
        expert_impact.index = ['No Expert', 'Has Expert']
        print("\nExpert Presence:")
        print(expert_impact)
    else:
        print("\n  No expert reviewer data available")
    
    # Shallow Reviewer 比例的影响
    if df['total_reviewers'].sum() > 0:
        df['shallow_ratio'] = df.apply(
            lambda x: x['num_shallow'] / x['total_reviewers'] if x['total_reviewers'] > 0 else 0, 
            axis=1
        )
        print("\nShallow Reviewer Ratio Impact:")
        for threshold in [0, 0.33, 0.5]:
            subset = df[df['shallow_ratio'] > threshold]
            if len(subset) > 0:
                print(f"  Shallow Ratio > {threshold:.0%}: Accept Rate = {subset['is_accepted'].mean():.1%} (n={len(subset)})")
    
    # 2.5 "High Credibility Low Score" 的杀伤力
    print("\n[2.5] The 'Expert Killer' Effect")
    print("-" * 80)
    if df['num_high_credibility_low_score'].sum() > 0:
        killer_impact = df.groupby(df['num_high_credibility_low_score'] > 0)['is_accepted'].agg(['mean', 'count'])
        killer_impact.index = ['No Expert Killer', 'Has Expert Killer (≥1)']
        print(killer_impact)
    else:
        print("  No 'High Credibility Low Score' reviewer data available")

    # ==========================================
    # SECTION 3: 分数统计特征分析
    # ==========================================
    print_section_header("SECTION 3: Score Statistics Analysis")
    
    analyze_score_distribution(df, "[3.1] Score Distribution by Decision")
    
    # 3.2 分数方差的影响（分歧度）
    print("\n[3.2] Score Variance (Reviewer Disagreement) Impact")
    print("-" * 80)
    
    valid_variance = df[df['avg_reviewer_score'] > 0]
    if len(valid_variance) > 0:
        median_variance = valid_variance['score_variance'].median()
        valid_variance = valid_variance.copy()
        valid_variance['high_variance'] = valid_variance['score_variance'] > median_variance
        variance_impact = valid_variance.groupby('high_variance')['is_accepted'].agg(['mean', 'count'])
        variance_impact.index = ['Low Variance (Consensus)', 'High Variance (Polarized)']
        print(variance_impact)
    else:
        print("  No valid score variance data")
    
    # 3.3 最低分的致命性
    print("\n[3.3] The 'Min Score' Death Threshold")
    print("-" * 80)
    
    valid_scores_df = df[df['min_reviewer_score'] > 0]
    if len(valid_scores_df) > 0:
        for threshold in [3, 4, 5, 6]:
            low_score = valid_scores_df[valid_scores_df['min_reviewer_score'] <= threshold]
            high_score = valid_scores_df[valid_scores_df['min_reviewer_score'] > threshold]
            if len(low_score) > 0 and len(high_score) > 0:
                print(f"Min Score ≤ {threshold}: Accept Rate = {low_score['is_accepted'].mean():.1%} (n={len(low_score)})")
                print(f"Min Score > {threshold}: Accept Rate = {high_score['is_accepted'].mean():.1%} (n={len(high_score)})")
                print()
    else:
        print("  No valid min score data")

    # ==========================================
    # SECTION 4: Rebuttal 策略深度分析
    # ==========================================
    print_section_header("SECTION 4: Rebuttal Strategy Analysis")
    
    analyze_acceptance_by_category(df, 'rebuttal_strength',
        "[4.1] Rebuttal Effectiveness vs Acceptance")
    
    # 4.2 Success Rate 的阈值效应
    print("\n[4.2] Rebuttal Success Rate Threshold Analysis")
    print("-" * 80)
    
    valid_success = df[df['success_rate'] > 0]
    if len(valid_success) > 0:
        for threshold in [0.5, 0.7, 0.9]:
            high_success = valid_success[valid_success['success_rate'] >= threshold]
            if len(high_success) > 0:
                print(f"Success Rate ≥ {threshold:.0%}: Accept Rate = {high_success['is_accepted'].mean():.1%} (n={len(high_success)})")
    else:
        print("  No valid success rate data (all values are 0 or missing)")
    
    # 4.3 致命的自我破坏行为
    print("\n[4.3] Author Self-Sabotage Analysis")
    print("-" * 80)
    
    sabotage_cases = df[df['author_self_sabotage'] == True]
    fatal_error_cases = df[df['admitted_fatal_error'] == True]
    
    if len(sabotage_cases) > 0:
        print(f"Papers with Self-Sabotage: Accept Rate = {sabotage_cases['is_accepted'].mean():.1%} (n={len(sabotage_cases)})")
    else:
        print("Papers with Self-Sabotage: No cases found")
    
    if len(fatal_error_cases) > 0:
        print(f"Papers admitting Fatal Error: Accept Rate = {fatal_error_cases['is_accepted'].mean():.1%} (n={len(fatal_error_cases)})")
    else:
        print("Papers admitting Fatal Error: No cases found")
    
    # 4.4 Reviewer State Transitions
    print("\n[4.4] Most Valuable Rebuttal Outcomes")
    print("-" * 80)
    
    state_columns = {
        'Converted_From_Low': 'count_converted_from_low',
        'Softened': 'count_softened',
        'Ghosted_But_Overruled': 'count_ghosted_but_overruled',
    }
    
    found_any = False
    for state, col in state_columns.items():
        if col in df.columns:
            with_state = df[df[col] > 0]
            if len(with_state) > 0:
                found_any = True
                print(f"{state:30s}: {with_state['is_accepted'].mean():.1%} (n={len(with_state)})")
    
    if not found_any:
        print("  No reviewer state transition data available")

    # ==========================================
    # SECTION 5: Decision Archetype 分析
    # ==========================================
    print_section_header("SECTION 5: Decision Archetype Patterns")
    
    analyze_acceptance_by_category(df, 'decision_archetype',
        "[5.1] Decision Archetype Distribution")
    
    print("\n[5.2] Common Archetypes in Rejected Papers")
    print("-" * 80)
    rejected = df[df['ground_truth'] == 'reject']
    if len(rejected) > 0:
        valid_rejected = rejected[rejected['decision_archetype'].notna() & (rejected['decision_archetype'] != 'Unknown')]
        if len(valid_rejected) > 0:
            print(valid_rejected['decision_archetype'].value_counts(normalize=True).head(5).round(3))
        else:
            print("  No valid archetype data for rejected papers")
    else:
        print("  No rejected papers found")

    # ==========================================
    # SECTION 6: 相关性矩阵
    # ==========================================
    print_section_header("SECTION 6: Feature Importance (Correlation Analysis)")
    
    numeric_features = [
        'avg_reviewer_score',
        'min_reviewer_score',
        'max_reviewer_score',
        'score_variance',
        'score_range',
        'success_rate',
        'num_experts',
        'num_shallow',
        'count_converted_from_low',
        'count_explicitly_stubborn',
        'is_accepted'
    ]
    
    analyze_correlation_matrix(df, numeric_features, 
        "[6.1] Numeric Feature Correlations with Acceptance")

    # ==========================================
    # SECTION 7: 反直觉发现
    # ==========================================
    print_section_header("SECTION 7: Counter-Intuitive Findings")
    
    print("\n[7.1] High-Score Rejections (Avg ≥ 6.5)")
    print("-" * 80)
    high_score_reject = df[(df['avg_reviewer_score'] >= 6.5) & (df['ground_truth'] == 'reject')]
    if len(high_score_reject) > 0:
        print(f"Count: {len(high_score_reject)}")
        
        valid_flaws = high_score_reject[high_score_reject['fatal_flaw'].notna() & (high_score_reject['fatal_flaw'] != 'None')]
        if len(valid_flaws) > 0:
            print("\nCommon Fatal Flaws:")
            print(valid_flaws['fatal_flaw'].value_counts().head(3))
        
        valid_risk = high_score_reject[high_score_reject['risk_level'].notna() & (high_score_reject['risk_level'] != 'Unknown')]
        if len(valid_risk) > 0:
            print("\nCommon Risk Levels:")
            print(valid_risk['risk_level'].value_counts())
    else:
        print("  No high-score rejections found (this is good!)")
    
    print("\n[7.2] Low-Score Acceptances (Avg < 5.5)")
    print("-" * 80)
    low_score_accept = df[(df['avg_reviewer_score'] < 5.5) & (df['avg_reviewer_score'] > 0) & (df['is_accepted'] == 1)]
    if len(low_score_accept) > 0:
        print(f"Count: {len(low_score_accept)}")
        
        valid_factors = low_score_accept[low_score_accept['key_factor'].notna() & (low_score_accept['key_factor'] != 'Unknown')]
        if len(valid_factors) > 0:
            print("\nKey Success Factors:")
            print(valid_factors['key_factor'].value_counts().head(3))
        
        valid_rebuttal = low_score_accept[low_score_accept['rebuttal_strength'].notna() & (low_score_accept['rebuttal_strength'] != 'None')]
        if len(valid_rebuttal) > 0:
            print("\nRebuttal Strength Distribution:")
            print(valid_rebuttal['rebuttal_strength'].value_counts())
    else:
        print("  No low-score acceptances found")

    # ==========================================
    # SECTION 8: 作者行动指南
    # ==========================================
    print_section_header("SECTION 8: Actionable Insights for Authors")
    
    print("\n[8.1] The 'Golden Trio' for Acceptance")
    print("-" * 80)
    
    df['has_expert_champion'] = (df['num_experts'] > 0) & (df['expert_score'].fillna(0) >= 7)
    df['strong_rebuttal'] = df['rebuttal_strength'].isin(['Strong', 'Moderate'])
    df['high_novelty'] = df['novelty'].isin(['Groundbreaking', 'Significant'])
    
    golden_trio = df[df['has_expert_champion'] & df['strong_rebuttal'] & df['high_novelty']]
    if len(golden_trio) > 0:
        print(f"Papers with all 3 factors: Accept Rate = {golden_trio['is_accepted'].mean():.1%} (n={len(golden_trio)})")
    else:
        print("Papers with all 3 factors: No papers found with all three")
    
    for factor, col in [
        ('Expert Champion', 'has_expert_champion'),
        ('Strong Rebuttal', 'strong_rebuttal'),
        ('High Novelty', 'high_novelty')
    ]:
        missing = df[~df[col]]
        if len(missing) > 0:
            print(f"Papers without {factor}: Accept Rate = {missing['is_accepted'].mean():.1%} (n={len(missing)})")
    
    print("\n[8.2] Red Flags to Avoid at All Costs")
    print("-" * 80)
    
    red_flags = [
        ('Admitting Fatal Error', df['admitted_fatal_error'] == True),
        ('Self-Sabotage in Rebuttal', df['author_self_sabotage'] == True),
        ('New Red Flag Raised', df['count_new_red_flag_raised'] > 0),
        ('Expert Gives Low Score (≤4)', (df['has_expert']) & (df['expert_score'].fillna(10) <= 4))
    ]
    
    for flag_name, condition in red_flags:
        flagged = df[condition]
        if len(flagged) > 0:
            print(f"{flag_name:40s}: Accept Rate = {flagged['is_accepted'].mean():.1%} (n={len(flagged)})")
        else:
            print(f"{flag_name:40s}: No cases found")

    print("\n[8.3] When to Fight vs When to Accept Rejection")
    print("-" * 80)
    
    worth_fighting = df[
        (df['novelty'].isin(['Groundbreaking', 'Significant'])) &
        (df['min_reviewer_score'] >= 4) &
        (df['count_new_red_flag_raised'] == 0)
    ]
    
    hopeless = df[
        (df['min_reviewer_score'] < 3) & (df['min_reviewer_score'] > 0) |
        (df['count_new_red_flag_raised'] > 0) |
        ((df['has_expert']) & (df['expert_score'].fillna(10) < 4))
    ]
    
    if len(worth_fighting) > 0:
        print(f"'Worth Fighting' Cases: Accept Rate = {worth_fighting['is_accepted'].mean():.1%} (n={len(worth_fighting)})")
    else:
        print("'Worth Fighting' Cases: No papers match criteria")
    
    if len(hopeless) > 0:
        print(f"'Hopeless' Cases: Accept Rate = {hopeless['is_accepted'].mean():.1%} (n={len(hopeless)})")
    else:
        print("'Hopeless' Cases: No papers match criteria")

    # ==========================================
    # SECTION 9: 模型预测性能分析
    # ==========================================
    print_section_header("SECTION 9: Model Prediction Performance")
    
    print("\n[9.1] Prediction Accuracy by Model")
    print("-" * 80)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()
        
        model_df['pred_binary'] = model_df['pred_decision'].apply(
            lambda x: 'accept' if x in ['oral', 'spotlight', 'poster'] else 'reject'
        )
        model_df['truth_binary'] = model_df['ground_truth'].apply(
            lambda x: 'accept' if x in ['oral', 'spotlight', 'poster'] else 'reject'
        )
        
        accuracy = (model_df['pred_binary'] == model_df['truth_binary']).mean()
        print(f"{model:20s}: Accuracy = {accuracy:.1%} (n={len(model_df)})")

    # ==========================================
    # 额外：导出样本数据用于调试
    # ==========================================
    print("\n" + "=" * 80)
    print("  SAMPLE DATA FOR VERIFICATION")
    print("=" * 80)
    
    print("\n[Sample of Key Fields - First 5 Papers]")
    sample_cols = ['title', 'ground_truth', 'novelty', 'consensus', 'fatal_flaw', 'rebuttal_strength', 'avg_reviewer_score']
    available_cols = [c for c in sample_cols if c in df.columns]
    print(df[available_cols].head().to_string())

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()