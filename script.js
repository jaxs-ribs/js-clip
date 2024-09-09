import { AutoTokenizer, CLIPTextModelWithProjection, AutoProcessor, CLIPVisionModelWithProjection, RawImage, cos_sim, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.8.0';

async function checkWebGPUSupport() {
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported on this browser.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('Couldn\'t request WebGPU adapter.');
    }
    return true;
}

function configureEnvironment(useWebGPU) {
    if (useWebGPU) {
        env.backends.onnx.wasm.numThreads = 1;
        env.backends.onnx.wasm.proxy = true;
        env.useBrowserCache = false;
        env.useWebGPU = true;
        env.allowLocalModels = false;
    } else {
        env.useWebGPU = false;
        env.backends.onnx.wasm.numThreads = 4;
    }
}

async function loadModels() {
    const modelName = 'Xenova/clip-vit-base-patch32';
    return {
        tokenizer: await AutoTokenizer.from_pretrained(modelName, { quantized: true }),
        text_model: await CLIPTextModelWithProjection.from_pretrained(modelName, { quantized: true }),
        processor: await AutoProcessor.from_pretrained(modelName, { quantized: true }),
        vision_model: await CLIPVisionModelWithProjection.from_pretrained(modelName, { quantized: true })
    };
}

async function getTextEmbedding(tokenizer, text_model, text) {
    const inputs = tokenizer([text], { padding: true, truncation: true });
    const { text_embeds } = await text_model(inputs);
    return text_embeds[0];
}

async function getImageEmbedding(processor, vision_model, imageUrl) {
    const image = await RawImage.read(imageUrl);
    const inputs = await processor(image);
    const { image_embeds } = await vision_model(inputs);
    return image_embeds[0];
}

async function processSearchableItems(tokenizer, text_model, processor, vision_model, items) {
    return Promise.all(items.map(async item => {
        if (item.text && item.image) {
            return {
                text_embed: await getTextEmbedding(tokenizer, text_model, item.text),
                image_embed: await getImageEmbedding(processor, vision_model, item.image),
                type: 'both'
            };
        } else if (item.text) {
            return {
                text_embed: await getTextEmbedding(tokenizer, text_model, item.text),
                type: 'text'
            };
        } else if (item.image) {
            return {
                image_embed: await getImageEmbedding(processor, vision_model, item.image),
                type: 'image'
            };
        }
    }));
}

function findBestMatches(query_embed, item_embeds, top_k = 3) {
    const similarities = item_embeds.map(item => {
        if (item.type === 'both') {
            return Math.max(
                cos_sim(query_embed.data, item.text_embed.data),
                cos_sim(query_embed.data, item.image_embed.data)
            );
        } else if (item.type === 'text') {
            return cos_sim(query_embed.data, item.text_embed.data);
        } else {
            return cos_sim(query_embed.data, item.image_embed.data);
        }
    });
    
    return similarities
        .map((similarity, index) => ({ index, similarity }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, top_k);
}

function generateResultHTML(query, items, matches) {
    let html = `<h3>Query: "${query}"</h3>`;
    matches.forEach(({ index, similarity }, i) => {
        const item = items[index];
        html += `
            <div class="result-item">
                <h4>Match ${i + 1}</h4>
                ${item.text ? `<p><strong>Text:</strong> "${item.text}"</p>` : ''}
                ${item.image ? `<p><strong>Image:</strong></p><img src="${item.image}" alt="Result image" style="max-width: 200px;">` : ''}
                <p><strong>Similarity:</strong> ${(similarity * 100).toFixed(2)}%</p>
            </div>
            ${i < matches.length - 1 ? '<hr>' : ''}
        `;
    });
    return html + '<hr class="query-separator">';
}

async function runDemo() {
    const output = document.getElementById('output');
    output.innerHTML = 'Initializing...<br>';

    try {
        const useWebGPU = await checkWebGPUSupport();
        configureEnvironment(useWebGPU);
        output.innerHTML += 'WebGPU supported. Loading models...<br>';
    } catch (error) {
        console.warn('WebGPU not available:', error);
        output.innerHTML += `WebGPU not available: ${error.message}. Falling back to CPU.<br>`;
        configureEnvironment(false);
    }

    try {
        env.allowLocalModels = false;  // Force using remote models
        const { tokenizer, text_model, processor, vision_model } = await loadModels();
        output.innerHTML += 'Models loaded. Processing...<br>';

        const searchableItems = [
            { text: 'A cat lounging in the sun', image: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba' },
            { text: 'A dog playing in the park', image: 'https://images.unsplash.com/photo-1543466835-00a7907e9de1' },
            { text: 'A colorful parrot perched on a branch', image: 'https://images.unsplash.com/photo-1552728089-57bdde30beb3' },
            { text: 'Some goldfish', image: 'https://images.unsplash.com/photo-1524704654690-b56c05c78a00' },
            { text: 'A majestic lion in the savanna', image: 'https://images.unsplash.com/photo-1614027164847-1b28cfe1df60' }
        ];

        const queries = ['A pet animal', 'A bird in nature', 'An aquatic creature'];

        const item_embeds = await processSearchableItems(tokenizer, text_model, processor, vision_model, searchableItems);

        output.innerHTML += '<h2>Text Query to Mixed Item Matching</h2>';
        for (const query of queries) {
            const query_embed = await getTextEmbedding(tokenizer, text_model, query);
            const matches = findBestMatches(query_embed, item_embeds, 3); // Ensure top 3 results
            output.innerHTML += generateResultHTML(query, searchableItems, matches);
        }

    } catch (error) {
        console.error('An error occurred:', error);
        output.innerHTML += `An error occurred: ${error.message}<br>`;
    }
}

console.log('Script loaded, calling runDemo');
runDemo();