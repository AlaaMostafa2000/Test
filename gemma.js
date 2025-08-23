import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import readline from 'readline';
import { pipeline } from '@xenova/transformers';

async function combineTextFiles(dir, outputFile) {
  const entries = await fs.readdir(dir);
  const files = entries.filter(f => f.endsWith('.txt'));
  let combined = '';
  for (const file of files) {
    const data = await fs.readFile(path.join(dir, file), 'utf8');
    combined += data + '\n';
  }
  await fs.writeFile(outputFile, combined);
  return outputFile;
}

export async function fineTuneModel(modelName, dataDir, outDir) {
  await fs.mkdir(outDir, { recursive: true });
  const datasetPath = path.join(outDir, 'dataset.txt');
  await combineTextFiles(dataDir, datasetPath);

  await new Promise((resolve, reject) => {
    const proc = spawn('python', ['scripts/train_gemma.py', modelName, datasetPath, outDir], { stdio: 'inherit' });
    proc.on('close', code => {
      if (code === 0) resolve();
      else reject(new Error(`Training failed with code ${code}`));
    });
  });
}

export async function chatWithModel(modelPath) {
  const generator = await pipeline('text-generation', modelPath, { quantized: true });
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const ask = () => {
    rl.question('You: ', async (prompt) => {
      if (prompt.trim().toLowerCase() === 'quit') {
        rl.close();
        return;
      }
      const stream = await generator(prompt, { max_new_tokens: 128, stream: true });
      for await (const chunk of stream) {
        const text = chunk.generated_text ?? chunk;
        process.stdout.write(text);
      }
      process.stdout.write('\n');
      ask();
    });
  };

  ask();
}

if (process.argv[1].endsWith('gemma.js')) {
  const mode = process.argv[2];
  if (mode === 'train') {
    const modelName = process.argv[3] || 'google/gemma-2-2b-it';
    const dataDir = process.argv[4] || './data';
    const outDir = process.argv[5] || './finetuned-gemma';
    fineTuneModel(modelName, dataDir, outDir).catch(err => {
      console.error(err);
      process.exit(1);
    });
  } else if (mode === 'chat') {
    const modelPath = process.argv[3] || './finetuned-gemma';
    chatWithModel(modelPath).catch(err => {
      console.error(err);
      process.exit(1);
    });
  } else {
    console.log('Usage:\n  node gemma.js train <modelName> <dataDir> <outDir>\n  node gemma.js chat <modelPath>');
  }
}
