const csv = require('csv-parser');
const fs = require('fs');
const fetch = require('node-fetch');
const results = [];

fs.createReadStream('../../../models.csv')
    .pipe(csv({
        headers: false
    }))
    .on('data', (data) => results.push(data))
    .on('end', () => {
        results.forEach((data) => {
            const name = data['0'];
            const url = data['1'];

            const root = `./static/${name}/`;

            fs.mkdirSync(root, { recursive: true });
            ["model.json", "group1-shard1of1.bin"].forEach((filename) => {
                const file_stream = fs.createWriteStream(`${root}${filename}`);

                fetch(`${url}tensorflowjs_model/${filename}`)
                    .then((response) => {
                        response.body.pipe(file_stream);
                    });
            });
        });
    });