import random
import uuid

from src.util.build_subsets import powerset
import math

from src.util.loader import build_log_from_subset

random.seed(26)

class Responsibility:
    def __init__(self, log, is_ocel:bool,
                 analysis_function=None,
                 event_attr="EventId", context_attr="case:concept:name",
                 constraint_attr:str="constraint_groups",
                 constraint_groups:list=None):
        # we assume xes log or flattened ocel log
        self.log = log
        self.is_ocel = is_ocel
        self.analysis_function = analysis_function
        self.events = set(log[event_attr])
        self.contexts = set(log[context_attr])
        self.event_attr = event_attr
        self.context_attr = context_attr
        self.constraint_attr = constraint_attr

        if constraint_groups is not None:
            self.__augment_with_constraint_attr(constraint_groups)

    def __compute_simple_value_for_item(self, players, item, coefficient_function, granularity="event"):
        num_players = len(players)
        other_players = [p for p in players if p != item]
        sum = 0

        for i, subset in enumerate(powerset(other_players)):
            #print(i)
            num_subset = len(subset)
            subset_of_items = set(subset)
            sublog_wo_item = self.__compute_sublog_for_simple_value(subset_of_items, granularity)
            subset_of_items.add(item)
            sublog_w_item = self.__compute_sublog_for_simple_value(subset_of_items, granularity)
            marg_contribution = self.__compute_marginal_contribution(sublog_wo_item, sublog_w_item)
            coefficient = coefficient_function(num_players, num_subset)
            sum += coefficient * marg_contribution

        return sum

    def __augment_with_constraint_attr(self, constraint_groups:list):
        self.log[self.constraint_attr] = None

        grouped_events = set()
        for constraint_group in constraint_groups:
            group_id = uuid.uuid4()
            self.log.loc[self.log[self.event_attr].isin(constraint_group), self.constraint_attr] = group_id
            grouped_events.update(constraint_group)

        for event in self.events - grouped_events:
            self.log.loc[self.log[self.event_attr] == event, self.constraint_attr] = uuid.uuid4()

    def __assert_constraint_groups_within_single_context(self):
        assert self.log.groupby(self.constraint_attr)[self.context_attr].nunique().max() <= 1, \
            "constrained nested values require each constraint group to lie within a single context"

    def compute_simple_values(self, granularity="event", value="shapley", constrained:bool = False):
        assert granularity in ["event", "context"]
        assert value in ["shapley", "banzhaf"]

        tmp_event_attr = self.event_attr
        if constrained:
            assert granularity == "event"
            self.event_attr = self.constraint_attr
            players = set(self.log[self.constraint_attr].unique())
        else:
            players = list(self.events.copy()) if granularity == "event" else list(self.contexts.copy())

        coefficient_function = Responsibility.__compute_shapley_coefficient if value == "shapley" else Responsibility.__compute_banzhaf_coefficient

        values: dict = {p: 0.0 for p in players}
        for p in players:
            values[p] = self.__compute_simple_value_for_item(players, p, coefficient_function, granularity)

        self.event_attr = tmp_event_attr
        return values

    def compute_simple_value_sampling(self, sample_size:int, granularity="event", value="shapley", antithetic=False, constrained:bool = False):
        cache:dict = dict() # keep set representations

        if antithetic:
            sample_size = int(sample_size / 2)

        tmp_event_attr = self.event_attr
        if constrained:
            assert granularity == "event"
            self.event_attr = self.constraint_attr
            players = list(self.log[self.constraint_attr].unique())
        else:
            players = list(self.events.copy()) if granularity == "event" else list(self.contexts.copy())

        values: dict = {p: 0.0 for p in players}  # keep value per player

        def cached_value(permutation):
            coalition_set = frozenset(permutation)
            if coalition_set not in cache:
                sublog = self.__compute_sublog_for_simple_value(coalition_set, granularity)
                cache[coalition_set] = self.analysis_function(sublog)
            return cache[coalition_set]

        if value == "shapley":
            num_iterations = int(sample_size / 2) if antithetic else sample_size
            sample_size = 2*num_iterations if antithetic else sample_size

            for i in range(1, num_iterations + 1):
                #print(i)
                perm = players[:]
                random.shuffle(perm)
                perms = [perm, list(reversed(perm))] if antithetic else [perm]

                for perm in perms:
                    coalition = []
                    prev_value = cached_value(coalition)
                    for player in perm:
                        coalition.append(player)
                        new_value = cached_value(coalition)
                        values[player] += new_value - prev_value
                        prev_value = new_value
        else:
            num_iterations = int(sample_size / 2) if antithetic else sample_size
            sample_size = 2 * num_iterations if antithetic else sample_size

            for _ in range(num_iterations):
                for player in players:
                    others = {p for p in players if p != player}
                    # Include each other player independently with probability 0.5
                    subset = {p for p in others if random.random() < 0.5}

                    it = [subset]

                    if antithetic:
                        complement = others - subset
                        it.append(complement)

                    for s in it:
                        value_without = cached_value(s)
                        s.add(player)
                        value_with = cached_value(s)

                        values[player] += value_with - value_without

        self.event_attr = tmp_event_attr
        return {p: values[p] / sample_size for p in players}

    def __compute_nested_value_for_item(self, item, coefficient_function, sample:int = 0):
        # find context id the event is contained in
        ci = self.log.loc[self.log[self.event_attr] == item, self.context_attr].iloc[0]

        # then consider
        events_in_ci = set(self.log.loc[self.log[self.context_attr] == ci, self.event_attr])
        events_in_ci.remove(item)

        other_contexts = self.contexts - {ci}

        sum = 0
        for subset_of_contexts in powerset(other_contexts, sample):
            contexts_incl_own = list(subset_of_contexts) + [ci]

            for item_subset in powerset(events_in_ci, sample):
                # 3 x um die Ecke denken - wir haben events_in_ci ohne item, aber retainen wir in dem fall nciht die items, die wir als set übergeben
                # wann ist dann item enthalten und wann nicht?
                filter_events = list(events_in_ci - set(item_subset))
                sublog_w_item = build_log_from_subset(self.log, self.is_ocel, filter_events, contexts_incl_own, self.event_attr)

                filter_events.append(item)
                sublog_wo_item = build_log_from_subset(self.log, self.is_ocel, filter_events, contexts_incl_own, self.event_attr)

                sum += self.__compute_marginal_contribution(sublog_wo_item, sublog_w_item) * \
                    coefficient_function(len(self.contexts), len(subset_of_contexts), len(events_in_ci) + 1, len(item_subset))

        return sum

    def compute_nested_value(self, granularity="event", value="owen", sample:int = 0, constrained:bool = False):
        assert granularity in ["event", "context"]
        assert value in ["owen", "banzhaf-owen"]

        if granularity == "context":
            simple_value = "shapley" if value == "owen" else "banzhaf"
            return self.compute_simple_values(granularity, simple_value, constrained)
        else:
            coefficient_function = self.__compute_owen_coefficient if value == "owen" else self.__compute_owen_banzhaf_coefficient

            tmp_event_attr = self.event_attr
            if constrained:
                self.__assert_constraint_groups_within_single_context()
                self.event_attr = self.constraint_attr
                players = set(self.log[self.constraint_attr].unique())
            else:
                players = self.events

            values: dict = {p: 0.0 for p in players}
            for p in players:
                values[p] = self.__compute_nested_value_for_item(p, coefficient_function, sample)

            self.event_attr = tmp_event_attr
            return values

    def compute_nested_value_sampling(self, sample_size:int, granularity="event", value="shapley", antithetic=False, constrained:bool = False):
        assert granularity in ["event"]
        assert value in ["owen", "banzhaf-owen"]

        cache:dict = dict() # keep set representations

        if antithetic:
            sample_size = int(sample_size / 2)

        tmp_event_attr = self.event_attr
        if constrained:
            self.__assert_constraint_groups_within_single_context()
            self.event_attr = self.constraint_attr
            players = list(self.log[self.constraint_attr].unique())
        else:
            players = list(self.events.copy())

        unions = list(self.contexts.copy())
        values: dict = {p: 0.0 for p in players}  # keep value per player

        def cached_value(permutation):
            coalition_set = frozenset(permutation)
            if coalition_set not in cache:
                sublog = self.__compute_sublog_for_simple_value(coalition_set, granularity)
                cache[coalition_set] = self.analysis_function(sublog)
            return cache[coalition_set]

        if value == "owen":
            num_iterations = sample_size
            for i in range(1, num_iterations + 1):
                perm_union = unions[:]
                random.shuffle(perm_union)
                perm = []

                for union in perm_union:
                    events_in_ci = list(set((self.log[self.log[self.context_attr] == union])[self.event_attr]))
                    random.shuffle(events_in_ci)
                    perm.extend(events_in_ci)

                perms = [perm, list(reversed(perm))] if antithetic else [perm]

                for perm in perms:
                    coalition = []
                    prev_value = cached_value(coalition)
                    for player in perm:
                        coalition.append(player)
                        new_value = cached_value(coalition)
                        values[player] += new_value - prev_value
                        prev_value = new_value

            if antithetic:
                sample_size = 2 * num_iterations
        else:
            for _ in range(sample_size):
                for player in players:
                    ci = self.log.loc[self.log[self.event_attr] == player, self.context_attr].iloc[0]
                    events_in_ci = set((self.log[self.log[self.context_attr] == ci])[self.event_attr])

                    other_unions = [u for u in unions if u != ci]
                    other_players_in_ci = [p for p in events_in_ci if p != player]
                    # Include each other player independently with probability 0.5
                    subset_unions = [u for u in other_unions if random.random() < 0.5]
                    subset_events= [p for p in other_players_in_ci if random.random() < 0.5]

                    # todo: without replacement - currently allowing for repetitions
                    events = []
                    for su in subset_unions:
                        events_in_su = set((self.log[self.log[self.context_attr] == su])[self.event_attr])
                        events.extend(events_in_su)
                    events.extend(subset_events)
                    value_without = cached_value(events)
                    value_with = cached_value(events + [player])

                    values[player] += value_with - value_without

        self.event_attr = tmp_event_attr
        return {p: values[p] / sample_size for p in players}


    def __compute_sublog_for_simple_value(self, subset, granularity="event"):
        if granularity == "event":
            sublog = build_log_from_subset(self.log, self.is_ocel, subset, None, self.event_attr)
        else:
            sublog = build_log_from_subset(self.log, self.is_ocel, None, subset, self.event_attr)

        return sublog

    def __compute_marginal_contribution(self, sublog_wo_item, sublog_w_item):
        v_wo_item = self.analysis_function(sublog_wo_item)
        v_w_item = self.analysis_function(sublog_w_item)

        return v_w_item - v_wo_item

    @staticmethod
    def __compute_shapley_coefficient(num_players:int, size_subset:int):
        a = math.factorial(size_subset)
        b = math.factorial(num_players - size_subset - 1)

        lower = math.factorial(num_players)

        return (a*b) / lower


    @staticmethod
    def __compute_banzhaf_coefficient(num_players:int, size_subset:int):
        return 1 / (math.pow(2, num_players - 1))

    @staticmethod
    def __compute_owen_coefficient(size_all_partitions, size_partition, size_set_of_item, size_subset):
        factor_partition =  (math.factorial(size_partition) * math.factorial(size_all_partitions - size_partition - 1)) / math.factorial(size_all_partitions)
        factor_set_of_item = (math.factorial(size_subset) * math.factorial(size_set_of_item - size_subset - 1)) / math.factorial(size_set_of_item)
        return factor_partition * factor_set_of_item

    @staticmethod
    def __compute_owen_banzhaf_coefficient(size_all_partitions, size_partition, size_set_of_item, size_subset):
        factor_partition = math.pow(2, size_all_partitions - 1)
        factor_subset = math.pow(2, size_set_of_item - 1)
        return 1 / (factor_partition * factor_subset)