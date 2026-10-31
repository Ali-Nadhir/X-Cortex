/* Browser regression checks against a running mock server. See README for usage. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const baseURL = process.env.XCORTEX_TEST_URL || 'http://127.0.0.1:8418';
const output = process.env.XCORTEX_SCREENSHOTS;
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO1cAAAAASUVORK5CYII=', 'base64');

(async () => {
  const browser = await chromium.launch({ headless: true, ...(process.env.XCORTEX_BROWSER_CHANNEL ? {channel: process.env.XCORTEX_BROWSER_CHANNEL} : {}) });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const check = name => console.log('PASS ' + name);
  const snap = async name => { if (output) { fs.mkdirSync(output, {recursive: true}); await page.screenshot({path:path.join(output, name+'.png'), fullPage:true}); } };
  try {
    const config = await (await context.request.get(baseURL + '/api/config')).json();
    assert.equal(config.mode, 'mock', 'Run browser checks only against an isolated mock server.');
    await page.goto(baseURL);
    await page.getByText('READY FOR STUDY', {exact:true}).waitFor();
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('#attachmentChip').isVisible(), false);
    assert.deepEqual(await page.locator('.welcome-bubble').boundingBox(), {x:542,y:292,width:814,height:382});
    assert(await page.evaluate(() => [...document.images].every(i => i.complete && i.naturalWidth > 0)));
    await snap('welcome-desktop');
    check('welcome geometry, local fonts and artwork');

    for (const name of ['Chats','Arrange chats','Case files and context','Models and tools','Settings']) {
      const button = page.getByRole('button', {name, exact:true});
      await button.click();
      assert.equal(await button.getAttribute('aria-expanded'), 'true');
      await page.keyboard.press('Escape');
      assert.equal(await button.getAttribute('aria-expanded'), 'false');
    }
    await page.getByRole('button',{name:'Settings',exact:true}).click();
    await page.locator('#densityToggle').click();
    assert(await page.locator('body').evaluate(el => el.classList.contains('compact')));
    await page.reload();
    await page.getByText('READY FOR STUDY', {exact:true}).waitFor();
    assert(await page.locator('body').evaluate(el => el.classList.contains('compact')));
    await page.getByRole('button',{name:'Settings',exact:true}).click();
    await page.locator('#densityToggle').click();
    await page.keyboard.press('Escape');
    check('five pop-outs, keyboard dismissal, persisted preferences');

    await page.locator('#fileInput').setInputFiles({name:'invalid.json',mimeType:'application/json',buffer:Buffer.from('{bad')});
    await page.getByText('Invalid JSON record: invalid.json', {exact:true}).waitFor();
    await page.locator('#fileInput').setInputFiles([
      {name:'ui-wrist.png',mimeType:'image/png',buffer:png},
      {name:'history.json',mimeType:'application/json',buffer:Buffer.from('{"age":72,"symptoms":"wrist pain"}')}
    ]);
    await page.locator('#attachmentName').filter({hasText:'history.json'}).waitFor();
    await page.locator('#historyInput').fill('Fall on left wrist. Please assess the image.');
    await page.getByRole('button',{name:'START ANALYSIS',exact:true}).click();
    await page.locator('.report').waitFor({timeout:30000});
    await page.getByText('CASE READY',{exact:true}).waitFor();
    assert.equal(await page.locator('.tool-card.complete').count(),3);
    const sessions = await (await context.request.get(baseURL+'/api/sessions')).json();
    const session = sessions.sessions.find(s => s.title === 'ui-wrist.png');
    const record = await (await context.request.get(baseURL+'/api/sessions/'+session.id)).json();
    assert(record.history.includes('wrist pain'));
    assert(record.history.includes('Fall on left wrist'));
    await snap('analysis-desktop');
    check('radiograph and JSON upload, real mock API stream, evidence and report');

    const overlay = page.locator('.viewer-overlay');
    await page.locator('.switch').uncheck();
    await page.locator('.ov-range').fill('80');
    assert.equal(await overlay.evaluate(el => el.style.opacity),'0');
    await page.locator('.switch').check();
    assert.equal(await overlay.evaluate(el => el.style.opacity),'0.8');
    await page.locator('#historyInput').fill('Summarize the evidence.');
    await page.locator('#historyInput').press('Enter');
    await page.getByText('CASE READY',{exact:true}).waitFor();
    await page.locator('.stream').last().filter({hasText:'Based on the session evidence'}).waitFor();
    check('follow-up conversation and overlay controls');

    await page.getByRole('button',{name:'Chats',exact:true}).click();
    await page.locator('#sessionSearch').fill('no-such-case');
    await page.getByText('No cases match your filters.',{exact:true}).waitFor();
    await page.locator('#sessionSearch').fill('ui-wrist');
    await page.locator('.pin-btn').first().click();
    assert.equal(await page.locator('.pin-btn').first().getAttribute('aria-pressed'),'true');
    await page.locator('.session').first().click();
    await page.locator('.report').waitFor();
    await page.getByRole('button',{name:'Case files and context',exact:true}).click();
    await page.locator('#contextFiles summary').click();
    assert((await page.locator('.context-history').textContent()).includes('history.json'));
    await snap('case-context');
    await page.keyboard.press('Escape');
    check('case search, pinning, reload, stored clinical context');

    await page.getByRole('button',{name:'Arrange chats',exact:true}).click();
    await page.getByRole('button',{name:'Body part',exact:true}).click();
    await page.getByRole('button',{name:'Chats',exact:true}).click();
    await page.locator('#sessionsList .eyebrow').filter({hasText:'WRIST'}).waitFor();
    await page.getByRole('button',{name:'Arrange chats',exact:true}).click();
    await page.locator('#hideCompleted').click();
    await page.getByRole('button',{name:'Chats',exact:true}).click();
    assert.equal(await page.locator('.session').count(),0);
    await page.getByRole('button',{name:'Arrange chats',exact:true}).click();
    await page.locator('#resetArrangement').click();
    await page.keyboard.press('Escape');
    await page.getByRole('button',{name:'Settings',exact:true}).click();
    await page.locator('#reportDetail').click();
    assert.equal(await page.locator('.report-limitations').isVisible(),true);
    assert.equal(await page.locator('.cases').isVisible(),false);
    await page.locator('#reportDetail').click();
    await page.keyboard.press('Escape');
    check('body-part grouping, completion filter and concise report retains limitations');

    await page.route('**/api/ask', route => route.fulfill({status:503,body:'Unavailable'}), {times:1});
    await page.locator('#historyInput').fill('Retry this question.');
    await page.locator('#sendBtn').click();
    await page.getByText('ATTENTION NEEDED',{exact:true}).waitFor();
    await page.getByRole('button',{name:'EDIT MESSAGE',exact:true}).click();
    assert.equal(await page.locator('#historyInput').inputValue(),'Retry this question.');
    await snap('error-recovery');
    await page.getByRole('button',{name:'RETRY',exact:true}).click();
    await page.getByText('CASE READY',{exact:true}).waitFor();
    assert.equal(await page.locator('.error-note').count(),0);
    check('HTTP error state, edit and successful retry');

    await page.route('**/api/ask', route => route.fulfill({status:200,contentType:'text/event-stream',body:'event: token\ndata: {"text":"Partial response"}\n\n'}), {times:1});
    await page.locator('#historyInput').fill('Interrupted stream');
    await page.locator('#sendBtn').click();
    await page.getByText('ATTENTION NEEDED',{exact:true}).waitFor();
    await page.getByText('The connection ended before the response completed.',{exact:true}).waitFor();
    check('truncated stream is reported as interrupted');

    await page.locator('#homeBtn').click();
    await page.locator('#fileInput').setInputFiles({name:'broken.png',mimeType:'image/png',buffer:Buffer.from('not an image')});
    await page.locator('#historyInput').fill('Preserve this history.');
    await page.locator('#sendBtn').click();
    await page.getByText('ATTENTION NEEDED',{exact:true}).waitFor();
    assert.equal(await page.locator('#historyInput').inputValue(),'Preserve this history.');
    assert((await page.locator('#attachmentName').textContent()).includes('broken.png'));
    check('failed upload retains image selection and clinical context');

    await page.locator('#homeBtn').click();
    for (const width of [390,768,1024,1440]) {
      await page.setViewportSize({width,height:900});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Horizontal overflow at '+width);
      assert(await page.locator('#sendBtn').isVisible());
      await page.getByRole('button',{name:'Settings',exact:true}).click();
      const box = await page.locator('#settingsPanel').boundingBox();
      assert(box.x + box.width <= width);
      await page.keyboard.press('Escape');
      if (width === 390) await snap('welcome-mobile');
    }
    check('responsive layout and pop-outs at 390, 768, 1024, 1440 pixels');

    await page.route('**/api/config', route => route.fulfill({status:503,body:'Offline'}), {times:1});
    await page.reload();
    await page.getByText('SERVER UNAVAILABLE',{exact:true}).waitFor();
    await page.getByRole('button',{name:'Models and tools',exact:true}).click();
    await page.locator('#reconnect').click();
    await page.getByText('READY FOR STUDY',{exact:true}).waitFor();
    check('offline startup and reconnection');
    assert.deepEqual(errors,[]);
    check('no browser JavaScript errors');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
