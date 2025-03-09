def extract_more_entities(news_df):
    """从新闻标题和内容中提取更多实体"""
    # 使用分类和子分类
    category_entities = set()
    
    # 添加从标题中提取关键词
    title_entities = set()
    
    # 使用简单的词频统计提取关键词
    from collections import Counter
    import re
    
    all_words = []
    for title in news_df['title']:
        if isinstance(title, str):
            # 简单分词
            words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
            all_words.extend(words)
    
    # 找出频率最高的词作为实体
    word_counter = Counter(all_words)
    common_words = [word for word, count in word_counter.most_common(1000) 
                   if count >= 5 and word not in STOP_WORDS]
    
    title_entities.update(common_words)
    
    # 合并所有实体
    all_entities = category_entities.union(title_entities)
    
    # 创建实体词典
    entity_list = sorted(list(all_entities))
    entity_dict = {entity: idx for idx, entity in enumerate(entity_list)}
    
    return entity_dict 