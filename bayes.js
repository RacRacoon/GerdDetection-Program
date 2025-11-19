// =================================================================
// 1. DATA MODEL YANG DIEKSPOR DARI PYTHON (Contoh Data)
//    PASTIKAN ANDA MENGGANTI NILAI INI DENGAN OUTPUT PYTHON ANDA
// =================================================================
const VOCABULARY_MAP = [
        "belum",
        "berbaring",
        "cepat",
        "dada",
        "dan",
        "dari",
        "di",
        "dimakan",
        "hati",
        "hebat",
        "ke",
        "kembali",
        "kembung",
        "kenyang",
        "makan",
        "makanan",
        "membungkuk",
        "mengakibatkan",
        "mual",
        "mulut",
        "naik",
        "nyeri",
        "pagi",
        "pahit",
        "panas",
        "perut",
        "saat",
        "sakit",
        "saya",
        "sedikit",
        "sehabis",
        "sekarang",
        "sesak",
        "setelah",
        "setiap",
        "telat",
        "terasa",
        "terbakar",
        "ulu",
        "yang"
    ];

    const CLASS_LOG_PRIOR = [
        -0.6931471805599452,
        -0.6931471805599452
    ];
    
    const FEATURE_LOG_PROB = [
        [
            -3.676300671907076,
            -4.3694478524670215,
            -3.676300671907076,
            -4.3694478524670215,
            -2.5776883832389665,
            -3.676300671907076,
            -3.676300671907076,
            -4.3694478524670215,
            -3.2708355637989115,
            -4.3694478524670215,
            -4.3694478524670215,
            -4.3694478524670215,
            -3.676300671907076,
            -3.676300671907076,
            -2.9831534913471307,
            -4.3694478524670215,
            -4.3694478524670215,
            -4.3694478524670215,
            -2.9831534913471307,
            -4.3694478524670215,
            -4.3694478524670215,
            -3.2708355637989115,
            -3.676300671907076,
            -4.3694478524670215,
            -4.3694478524670215,
            -3.676300671907076,
            -3.676300671907076,
            -3.676300671907076,
            -2.760009940032921,
            -3.676300671907076,
            -3.676300671907076,
            -3.676300671907076,
            -4.3694478524670215,
            -4.3694478524670215,
            -3.676300671907076,
            -3.676300671907076,
            -2.9831534913471307,
            -4.3694478524670215,
            -3.2708355637989115,
            -4.3694478524670215
        ],
        [
            -4.356708826689592,
            -3.6635616461296463,
            -4.356708826689592,
            -3.2580965380214817,
            -2.970414465569701,
            -4.356708826689592,
            -4.356708826689592,
            -3.6635616461296463,
            -4.356708826689592,
            -3.6635616461296463,
            -3.6635616461296463,
            -3.2580965380214817,
            -4.356708826689592,
            -4.356708826689592,
            -3.2580965380214817,
            -3.2580965380214817,
            -3.6635616461296463,
            -3.6635616461296463,
            -3.6635616461296463,
            -3.2580965380214817,
            -3.2580965380214817,
            -4.356708826689592,
            -4.356708826689592,
            -3.6635616461296463,
            -3.6635616461296463,
            -4.356708826689592,
            -3.2580965380214817,
            -4.356708826689592,
            -2.970414465569701,
            -4.356708826689592,
            -4.356708826689592,
            -4.356708826689592,
            -3.6635616461296463,
            -3.2580965380214817,
            -4.356708826689592,
            -4.356708826689592,
            -2.747270914255491,
            -3.6635616461296463,
            -4.356708826689592,
            -3.6635616461296463
        ]
    ];

// Peta label yang mudah dibaca (SAMA seperti SENTIMENT_MAP di Python)
const SENTIMENT_MAP = { 0: 'MAAG', 1: 'GERD' };

// =================================================================
// 2. FUNGSI LOGIKA CORE (Implementasi Naive Bayes)
// =================================================================

// Fungsi yang meniru CountVectorizer
function vectorizeInput(text, vocab) {
    // Tokenisasi: Ubah teks menjadi huruf kecil dan bagi menjadi kata-kata
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const vector = new Array(vocab.length).fill(0);

    // Hitung frekuensi kata berdasarkan vocabulary
    for (const word of words) {
        const index = vocab.indexOf(word);
        if (index !== -1) {
            vector[index]++;
        }
    }
    return vector;
}

// Fungsi Prediksi Naive Bayes (Mengimplementasikan P(K|X) ∝ P(K) * P(X|K))
function predict(inputVector) {
    let bestScore = -Infinity;
    let bestClass = -1;

    // Loop melalui setiap kelas (0=MAAG, 1=GERD)
    for (let classIndex = 0; classIndex < CLASS_LOG_PRIOR.length; classIndex++) {
        
        // Mulai dengan probabilitas prior (log P(K))
        let score = CLASS_LOG_PRIOR[classIndex];
        
        // Tambahkan probabilitas fitur (log P(X|K))
        for (let i = 0; i < inputVector.length; i++) {
            const count = inputVector[i];
            
            // Jika kata muncul (count > 0), tambahkan probabilitas log P(kata|Kelas) sebanyak count
            if (count > 0) {
                const logProb = FEATURE_LOG_PROB[classIndex][i];
                score += (count * logProb);
            }
        }

        // Cek apakah skor ini lebih baik (lebih tinggi)
        if (score > bestScore) {
            bestScore = score;
            bestClass = classIndex;
        }
    }

    return bestClass; // Mengembalikan 0 (MAAG) atau 1 (GERD)
}

// =================================================================
// 3. FUNGSI ANTARMUKA (DOM Manipulation)
// =================================================================

document.addEventListener('DOMContentLoaded', () => {
    const btn         = document.getElementById('analisisButton');
    const input       = document.getElementById('keluhanInput');

    // 1. Klik tombol → diagnosa (sudah ada)
    btn.addEventListener('click', lakukanDiagnosa);

    // 2. Tekan Enter di textarea → diagnosa (BARU!)
    input.addEventListener('keydown', function(e) {
        // Hanya aktif kalau tekan Enter TANPA Shift (biar bisa enter baris baru kalau mau)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();        // cegah enter bikin baris baru
            lakukanDiagnosa();         // langsung diagnosa
        }
    });

    // Pindahkan logika diagnosa ke fungsi terpisah supaya bisa dipakai dua-duanya
    function lakukanDiagnosa() {
        const keluhan = input.value.trim();

        if (!keluhan) {
            tampilkanHasil("Mohon tulis keluhanmu dulu ya");
            return;
        }

        const vector = vectorizeInput(keluhan, VOCABULARY_MAP);
        const kelas  = predict(vector);

        if (kelas === 0) {
            tampilkanHasil(`
                <strong style="font-size:1.3em; color:#92400e;">Kemungkinan besar MAAG</strong><br>
                Gejala umum: nyeri ulu hati • kembung • mual setelah makan<br><br>
                <strong>Rekomendasi Obat Awal:</strong><br>
                • Promag (1–2 tablet kunyah, 3–4x sehari sebelum makan)<br>
                • Mylanta (10 ml sirup, 3–4x sehari setelah makan)<br><br>
                <small style="color:#64748b;">Konsultasi dokter untuk dosis tepat!</small>
            `, "maag");
        } else {
            tampilkanHasil(`
                <strong style="font-size:1.3em; color:#1e40af;">Kemungkinan besar GERD</strong><br>
                Gejala khas: dada panas/terbakar • asam naik • mulut pahit • sering sendawa<br><br>
                <strong>Rekomendasi Obat Awal:</strong><br>
                • Polysilane (10 ml sirup, 3–4x sehari setelah makan)<br>
                • Omeprazole 20 mg (1-2x sebelum makan)<br><br>
                <small style="color:#64748b;">Angkat kepala tempat tidur & hindari makan larut malam!</small>
            `, "gerd");
        }
    }
});
