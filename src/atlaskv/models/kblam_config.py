from transformers import PretrainedConfig


class KBLaMConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_name_or_path: str = "",
        kb_layer_frequency: int = 3,
        kb_scale_factor: int | None = None,
        top_k_kb: int = 100,
        dynamic_sparsify: bool = False,
        sep_query_head: bool = False,
        attn_implementation: str = "eager",
        use_kg: bool = False,
        use_hierarchial_kv: bool = False,
        **kwargs,
    ):
        """
        Configurations For KBLaM
        """
        self.base_model_name_or_path = base_model_name_or_path
        self.kb_layer_frequency = kb_layer_frequency
        self.kb_scale_factor = kb_scale_factor
        self.top_k_kb = top_k_kb
        self.dynamic_sparsify = dynamic_sparsify
        self.sep_query_head = sep_query_head
        self.attn_implementation = attn_implementation
        self.use_kg = use_kg
        self.use_hierarchial_kv = use_hierarchial_kv
        super().__init__(**kwargs)


class AtlasKVConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_name_or_path: str = "",
        kb_layer_frequency: int = 3,
        kb_scale_factor: int | None = None,
        root_top_k_kb: int = 100,
        inter_top_k_kb: int = 100,
        leaf_top_k_kb: int = 100,
        dynamic_sparsify: bool = False,
        sep_query_head: bool = False,
        attn_implementation: str = "eager",
        use_kg: bool = True,
        use_hierarchial_kv: bool = False,
        **kwargs,
    ):
        """
        Configurations For AtlasKV
        """
        self.base_model_name_or_path = base_model_name_or_path
        self.kb_layer_frequency = kb_layer_frequency
        self.kb_scale_factor = kb_scale_factor
        self.root_top_k_kb = root_top_k_kb
        self.inter_top_k_kb = inter_top_k_kb
        self.leaf_top_k_kb = leaf_top_k_kb
        self.dynamic_sparsify = dynamic_sparsify
        self.sep_query_head = sep_query_head
        self.attn_implementation = attn_implementation
        self.use_kg = use_kg
        self.use_hierarchial_kv = use_hierarchial_kv
        super().__init__(**kwargs)