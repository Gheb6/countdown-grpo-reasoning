const fs = require('fs');
const { PDFDocument } = require('pdf-lib');

async function mergePdfs({ items, outputPath, docTitle }) {
  const mergedPdf = await PDFDocument.create();
  console.log(`\n================ Creating: ${outputPath} ================`);

  let currentSheet = 1;
  let totalPages = 0;

  for (let idx = 0; idx < items.length; idx++) {
    const item = items[idx];
    const isLastItem = (idx === items.length - 1);
    const fileBytes = fs.readFileSync(item.file);
    const srcDoc = await PDFDocument.load(fileBytes, { ignoreEncryption: true });
    const srcPageCount = srcDoc.getPageCount();

    // Determine which 0-based page indices to take
    let indices = [];
    if (item.pages === 'all') {
      indices = Array.from({ length: srcPageCount }, (_, i) => i);
    } else {
      indices = item.pages.map(p => p - 1); // convert 1-based to 0-based
    }

    console.log(`\nAdding: "${item.title}"`);
    console.log(`Source file: ${item.file}`);
    console.log(`Pages selected: ${indices.map(i => i + 1).join(', ')} (count: ${indices.length})`);
    console.log(`Starts on PDF page: ${totalPages + 1} (Sheet ${currentSheet} FRONTE)`);

    const copiedPages = await mergedPdf.copyPages(srcDoc, indices);
    for (const page of copiedPages) {
      mergedPdf.addPage(page);
      totalPages++;
    }

    // Check if odd number of pages: if so, add a blank page for duplex printing UNLESS it's the last paper
    if (copiedPages.length % 2 !== 0) {
      if (!isLastItem) {
        console.log(`-> Has odd number of pages (${copiedPages.length}). Inserting 1 BLANK PAGE for duplex alignment.`);
        const lastPage = copiedPages[copiedPages.length - 1];
        const { width, height } = lastPage.getSize();
        mergedPdf.addPage([width, height]);
        totalPages++;
        console.log(`-> Blank page added at PDF page ${totalPages} (Sheet ${currentSheet + Math.floor(copiedPages.length / 2)} RETRO)`);
      } else {
        console.log(`-> Has odd number of pages (${copiedPages.length}), but is the LAST paper: omitting final blank page.`);
      }
    } else {
      console.log(`-> Has even number of pages (${copiedPages.length}). Perfectly aligned for duplex.`);
    }

    const sheetsUsed = Math.ceil(copiedPages.length / 2);
    console.log(`-> Sheets used: ${sheetsUsed} (Sheets ${currentSheet} to ${currentSheet + sheetsUsed - 1})`);
    currentSheet += sheetsUsed;
  }

  const mergedBytes = await mergedPdf.save();
  fs.writeFileSync(outputPath, mergedBytes);
  console.log(`\nSuccessfully created: ${outputPath}`);
  console.log(`Total PDF Pages: ${totalPages}`);
}

async function run() {
  // 1. Dispensa con i soli estratti raccomandati per lo studio
  const estrattiItems = [
    {
      title: '1. DeepSeekMath - Sez. 4: GRPO & Reinforcement Learning',
      file: '2402.03300v3.pdf',
      pages: [11, 12, 13, 14, 15] // 5 pagine
    },
    {
      title: '2. DeepSeek-R1 - Sez. 2: R1-Zero & Verifiable Rewards',
      file: 'DeepSeek-R1_ Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf',
      pages: [1, 2, 3, 4, 5] // 5 pagine
    },
    {
      title: '3. Gandhi et al. - Countdown Task & Four Habits',
      file: 'Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs.pdf',
      pages: [1, 2, 3, 4, 5] // 5 pagine
    },
    {
      title: '4. Wei et al. - Chain-of-Thought Baseline & Arithmetic Reasoning',
      file: 'NeurIPS-2022-chain-of-thought-prompting-elicits-reasoning-in-large-language-models-Paper-Conference.pdf',
      pages: [1, 2, 3, 4, 5] // 5 pagine
    }
  ];

  await mergePdfs({
    items: estrattiItems,
    outputPath: 'Dispensa_Estratti_Studio_FronteRetro.pdf',
    docTitle: 'Dispensa Studio GRPO e Countdown'
  });

  // 2. Tutti i paper completi
  const completiItems = [
    {
      title: '1. DeepSeekMath (Completo)',
      file: '2402.03300v3.pdf',
      pages: 'all' // 22 pagine (pari)
    },
    {
      title: '2. DeepSeek-R1 (Completo)',
      file: 'DeepSeek-R1_ Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf',
      pages: 'all' // 8 pagine (pari)
    },
    {
      title: '3. Gandhi et al. (Completo)',
      file: 'Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs.pdf',
      pages: 'all' // 10 pagine (pari)
    },
    {
      title: '4. Wei et al. - Chain of Thought (Completo)',
      file: 'NeurIPS-2022-chain-of-thought-prompting-elicits-reasoning-in-large-language-models-Paper-Conference.pdf',
      pages: 'all' // 9 pagine (dispari, ma ultimo -> niente bianca finale)
    }
  ];

  await mergePdfs({
    items: completiItems,
    outputPath: 'Tutti_I_Paper_Completi_FronteRetro.pdf',
    docTitle: 'Raccolta Paper Completi'
  });
}

run().catch(console.error);
