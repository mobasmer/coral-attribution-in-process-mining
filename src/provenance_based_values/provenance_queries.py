class QueryBuilder:
    schema2activities = {
        "application": ("A_Validating","A_Pending"),
        "workflow": ("W_Handle leads", "W_Complete application"),
        "offer": ("O_Sent (mail and online)", "O_Cancelled"),
        "case_r": ("W_Handle leads", "W_Complete application")
    }

    def __init__(self, schema, granularity):
        self.schema = schema
        self.granularity = granularity

        self.activityA = self.schema2activities[schema][0]
        self.activityB = self.schema2activities[schema][1]

    def prepare_db(self):
        query_str = f"""
                    DROP EXTENSION IF EXISTS provsql CASCADE;
                    CREATE EXTENSION provsql CASCADE;
                    SET SEARCH_PATH TO {self.schema}, public, provsql;
                    """
        return query_str
        
    def add_provenance(self):
        query_str = f"""
                    SELECT provsql.add_provenance('{self.schema}.{self.granularity}');
                    """
        return query_str

    def add_provenance_mapping(self):    
        query_str = f"""
                    DROP TABLE IF EXISTS {self.granularity}_{self.granularity}_idx;
                    SELECT provsql.create_provenance_mapping('{self.granularity}_{self.granularity}_idx', '{self.schema}.{self.granularity}', '{self.granularity}_idx');
                    """
        return query_str
    
    def remove_provenance(self):
        query_str = f"""
                    SELECT provsql.remove_provenance('{self.schema}.{self.granularity}');
                    """
        return query_str

    def get_grouping(self):
        query_str = f"""
                    SELECT event_idx, c_id FROM {self.schema}.event e;
                    """
        return query_str

    def get_lineage_for_simple_dfg(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT DISTINCT e1.activity as src, e2.activity as dest
                            FROM {self.schema}.relation r JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                        ) t;
                    """
        return query_str

    def get_lineage_for_simple_efg(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT DISTINCT e1.activity as src, e2.activity as dest
                            FROM {self.schema}.event e1 
                                JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                JOIN {self.schema}.event e2 ON e1.timestamp < e2.timestamp 
                                JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                    WHERE c1.context_idx = c2.context_idx
                    ) t;
                    """
        return query_str

    def get_lineage_for_freq_efg(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(frequency, '{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT e1.activity as src, e2.activity as dest, COUNT(*) as frequency
                            FROM {self.schema}.event e1 
                                JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                JOIN {self.schema}.event e2 ON e1.timestamp < e2.timestamp 
                                JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                    WHERE c1.context_idx = c2.context_idx
                    GROUP BY e1.activity, e2.activity
                    ) t;
                    """
        return query_str

    def get_lineage_for_simple_handover(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT DISTINCT e1.resource_idx as src, e2.resource_idx as dest
                            FROM {self.schema}.relation r JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                            WHERE e1.resource_idx != e2.resource_idx AND e1.resource_idx IS NOT NULL AND e2.resource_idx IS NOT NULL
                        ) t;
                    """
        return query_str

    def get_lineage_for_freq_handover(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(frequency, '{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT e1.resource_idx as src, e2.resource_idx as dest, COUNT(*) as frequency
                            FROM {self.schema}.relation r JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                            WHERE e1.resource_idx != e2.resource_idx AND e1.resource_idx IS NOT NULL AND e2.resource_idx IS NOT NULL
                            GROUP BY src, dest
                        ) t;
                    """
        return query_str

    def get_lineage_for_conformance_response(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT DISTINCT e1.event_idx as src, e2.event_idx as dest
                            FROM {self.schema}.relation r JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                        ) t;
                    """
        return query_str

    def get_lineage_for_violation_alternate_response(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT DISTINCT e1.event_idx as src, e2.event_idx as dest
                            FROM {self.schema}.event e1
                                 JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                 JOIN {self.schema}.event e2 ON e2.timestamp > e1.timestamp AND e2.activity = '{self.activityB}'
                                 JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id AND c2.context_idx = c1.context_idx
                                 JOIN {self.schema}.event e3 ON e1.timestamp < e3.timestamp AND e3.timestamp < e2.timestamp AND e3.activity = '{self.activityA}'
                                 JOIN {self.schema}.context c3 ON c3.context_idx = e3.c_id AND c3.context_idx = c1.context_idx
                            WHERE e1.activity = '{self.activityA}'
                        ) t;
                    """
        return query_str

    def get_lineage_for_violation_alternate_precedence(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT DISTINCT e1.event_idx as src, e2.event_idx as dest
                            FROM {self.schema}.event e1
                                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                                     JOIN {self.schema}.event e2 ON e2.timestamp < e1.timestamp AND e2.activity = '{self.activityB}'
                                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id AND c2.context_idx = c1.context_idx
                                                     JOIN {self.schema}.event e3 ON e1.timestamp > e3.timestamp AND e3.timestamp > e2.timestamp AND e3.activity = '{self.activityA}'
                                                     JOIN {self.schema}.context c3 ON c3.context_idx = e3.c_id AND c3.context_idx = c1.context_idx
                            WHERE e1.activity = '{self.activityA}'
                        ) t;
                    """
        return query_str

        #query_str = f"""
        #            SELECT e1.activity as src, e2.activity as dest, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx')
        #            FROM {self.schema}.relation r JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
        #                                     JOIN {self.schema}.context c1 ON c1.context_id = e1.c_id
        #                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
        #                                     JOIN {self.schema}.context c2 ON c2.context_id = e2.c_id
        #            GROUP BY e1.activity, e2.activity;
        #            """

        return query_str
    
    def get_lineage_for_frequency_dfg(self):
        query_str = f"""
                    SELECT *, provsql.sr_formula(frequency, '{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (SELECT e1.activity as src, e2.activity as dest, COUNT(*) as frequency
                            FROM {self.schema}.relation r JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                            GROUP BY e1.activity, e2.activity) as t;
                    """
        return query_str
    
    def get_lineage_for_max_dfg(self):
        query_str = f"""
                    SELECT *, provsql.sr_formula(max_duration, '{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (SELECT e1.activity as src, e2.activity as dest, MAX(r.duration) as max_duration
                            FROM {self.schema}.relation r
                                     JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                            GROUP BY e1.activity, e2.activity) as t;
                    """
        return query_str

    def get_lineage_for_simple_dfg_w_view(self):
        query_str = f"""
                    SELECT t.src, t.dest, NULL::NUMERIC, provsql.sr_formula(provsql.provenance(),'{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (
                            SELECT DISTINCT e1.activity as src, e2.activity as dest
                            FROM {self.schema}.relation r JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                        ) t;
                    """

        return query_str

    def get_lineage_for_frequency_dfg_w_view(self):
        query_str = f"""
                    SELECT *, provsql.sr_formula(frequency, '{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (SELECT e1.activity as src, e2.activity as dest, COUNT(*) as frequency
                            FROM {self.schema}.relation r
                                     JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                            GROUP BY e1.activity, e2.activity) as t;
                    """
        return query_str

    def get_lineage_for_max_dfg_w_view(self):
        query_str = f"""
                    SELECT *, provsql.sr_formula(max_duration, '{self.granularity}_{self.granularity}_idx') AS formula
                    FROM (SELECT e1.activity as src, e2.activity as dest, MAX(r.duration) as max_duration
                            FROM {self.schema}.relation r
                                     JOIN {self.schema}.event e1 ON e1.event_idx = r.src_event_id
                                     JOIN {self.schema}.context c1 ON c1.context_idx = e1.c_id
                                     JOIN {self.schema}.event e2 ON e2.event_idx = r.dest_event_id
                                     JOIN {self.schema}.context c2 ON c2.context_idx = e2.c_id
                            GROUP BY e1.activity, e2.activity) as t;
                    """
        return query_str

