/**
 * Browser probe for SensaCine API — paste in DevTools Console
 *
 * HOW TO USE:
 * 1. Open https://www.sensacine.com in your browser
 * 2. Open DevTools (F12 or Cmd+Option+I)
 * 3. Go to Console tab
 * 4. Copy-paste this entire script and press Enter
 * 5. It will fetch showtimes for Yelmo Ideal and print the JSON structure
 *
 * You can also change THEATER_ID to test other cinemas.
 */

(async function probeSensaCineAPI() {
  const THEATER_ID = 'E0621'; // Yelmo Ideal
  const DATE = new Date().toISOString().split('T')[0]; // today

  console.log(`%c=== SensaCine API Probe ===`, 'font-weight:bold;font-size:14px');
  console.log(`Theater: ${THEATER_ID}, Date: ${DATE}`);

  let allResults = [];
  let page = 1;
  let totalPages = 1;

  while (page <= totalPages) {
    const url = `/_/showtimes/theater-${THEATER_ID}/d-${DATE}/p-${page}`;
    console.log(`Fetching: ${url}`);

    const resp = await fetch(url);
    if (!resp.ok) {
      console.error(`HTTP ${resp.status}: ${resp.statusText}`);
      return;
    }

    const data = await resp.json();
    totalPages = data.pagination?.totalPages || 1;
    const results = data.results || [];
    allResults.push(...results);
    console.log(`  Page ${page}/${totalPages}: ${results.length} movies`);
    page++;
  }

  // Summary
  console.log(`\n%c=== SUMMARY ===`, 'font-weight:bold');
  console.log(`Total movies: ${allResults.length}`);
  let totalShowtimes = 0;

  allResults.forEach(entry => {
    const title = entry.movie?.title || '?';
    const showtimes = entry.showtimes || {};
    let count = 0;
    Object.values(showtimes).forEach(arr => {
      if (Array.isArray(arr)) count += arr.length;
    });
    totalShowtimes += count;
    console.log(`  ${title}: ${count} sessions`);
  });
  console.log(`Total showtimes: ${totalShowtimes}`);

  // Print first movie structure for inspection
  if (allResults.length > 0) {
    console.log(`\n%c=== FIRST MOVIE STRUCTURE ===`, 'font-weight:bold');
    console.log(JSON.stringify(allResults[0], null, 2));
  }

  // Full response — copy this from console
  const fullData = { pagination: { page: 1, totalPages: 1 }, results: allResults };
  console.log(`\n%c=== FULL JSON (copy below) ===`, 'font-weight:bold;color:green');
  console.log(JSON.stringify(fullData));

  // Also save to clipboard if possible
  try {
    await navigator.clipboard.writeText(JSON.stringify(fullData, null, 2));
    console.log('%c JSON copied to clipboard!', 'color:green;font-weight:bold');
    console.log('Paste it into tests/fixtures/sensacine_sample.json');
  } catch (e) {
    console.log('Could not copy to clipboard. Right-click the JSON above → Copy string contents');
  }
})();
