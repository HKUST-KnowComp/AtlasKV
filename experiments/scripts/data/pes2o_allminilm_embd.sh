CLUSTER=False
GENERATING_EMBEDDINGS=True

if [ "$CLUSTER" = True ]; then
    CLUSTER_FLAG="--cluster"
else
    CLUSTER_FLAG=""
fi

if [ "$GENERATING_EMBEDDINGS" = True ]; then
    GENERATING_EMBEDDINGS_FLAG="--generating_embeddings"
else
    GENERATING_EMBEDDINGS_FLAG=""
fi

python generate_kb_embeddings_gmm.py \
	--dataset_name atlas_pes2o_qkv \
    --dataset_path your_output_path/atlas_pes2o_qkv.json \
    --output_path your_output_path \
    --model_name all-MiniLM-L6-v2 \
	--endpoint_url ***	\
	--endpoint_api_key *** \
	--max_concurrency 512 \
	$CLUSTER_FLAG \
	$GENERATING_EMBEDDINGS_FLAG