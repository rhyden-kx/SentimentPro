import axios from 'axios';
import { parse } from 'json2csv';
import fs from 'fs';

// Make a GET request to your localhost server
axios.get('http://localhost:3000/api/apps/sg.com.gxs.app/reviews/')
  .then(response => {
    const jsonData = response.data.results.data; // Extract data from the 'results' object

    // Check if there is data to convert
    if (jsonData.length === 0) {
      console.log('No data to convert to CSV.');
      return;
    }

    // Define the fields for CSV conversion
    const fields = [
      'id',
      'userName',
      'userImage',
      'date',
      'score',
      'scoreText',
      'url',
      'title',
      'text',
      'replyDate',
      'replyText',
      'version',
      'thumbsUp',
      'criterias'
    ];

    // Convert JSON data to CSV format
    const csvData = parse(jsonData, { fields });

    // Write CSV data to a file
    fs.writeFileSync('PlayStoreData.csv', csvData);
    console.log('CSV file created successfully.');
  })
  .catch(error => {
    console.error('Error fetching data:', error);
  });
