class document():
    def __init__(self, cp_ids_counts_map, human_label):
        self.cp_ids_counts = cp_ids_counts_map # dictionary {context_predicate_id -- count}
        self.human_label = human_label
        self.length = sum(self.cp_ids_counts.values()) 
        self.model_label = -1
    
    def __str__(self):
        _str = 'Document:\n\t Ids-Counts:'
        for cp_id in self.cp_ids_counts:
            _str += str(cp_id) + '-' + str(self.cp_ids_counts[cp_id]) + ' '
        _str += '\n\t human label :' + str(self.human_label)
        return _str