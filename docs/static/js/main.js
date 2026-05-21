// ---------- Hero intro: FLIP-style morph on first scroll ----------
//
// On scroll the veil's title morphs into the in-flow paper title, then the veil
// fades out. Because it ends exactly where the in-flow title lives, the final
// reveal feels continuous.
(function introMorph() {
  var html = document.documentElement;
  if (!html.classList.contains('intro-active')) return;

  var MORPH_MS = 950;
  var FADE_MS  = 350;

  var triggered = false;

  syncIntroTitleMetrics();
  playIntroVideos();
  requestAnimationFrame(syncIntroTitleMetrics);
  window.addEventListener('resize', syncIntroTitleMetrics, { passive: true });
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(syncIntroTitleMetrics);
  }

  function exit() {
    if (triggered) return;
    triggered = true;
    morph();
  }

  function morph() {
    var veilTitle = document.querySelector('#intro-veil .intro-title');
    var flowTitle = document.querySelector('.paper-title');

    // Fallback: if either title is missing, just fade the veil out.
    if (!veilTitle || !flowTitle) {
      html.classList.remove('intro-active');
      html.classList.add('intro-done');
      setTimeout(cleanup, FADE_MS);
      return;
    }

    syncIntroTitleMetrics();

    // In-flow targets — they have visibility:hidden but layout exists, so
    // getBoundingClientRect returns valid coordinates.
    var fTit = flowTitle.getBoundingClientRect();
    var vTit = veilTitle.getBoundingClientRect();

    var titDx = (fTit.left + fTit.width / 2)  - (vTit.left + vTit.width / 2);
    var titDy = (fTit.top  + fTit.height / 2) - (vTit.top  + vTit.height / 2);

    // Kill the entrance keyframe animations so they can't fight the transitions.
    veilTitle.style.animation = 'none';

    // Switch state: CSS now enables `transition: transform ...` on the title,
    // and starts the hint fade-out.
    html.classList.remove('intro-active');
    html.classList.add('intro-fading');

    // Force a reflow so the new transition rule is observed before we change
    // the transform value (otherwise the browser may collapse the change into
    // an instant jump).
    /* eslint-disable-next-line no-unused-expressions */
    veilTitle.offsetHeight;

    // Apply target transform. The title metrics are identical, so this is a
    // pure translate with no scale or rewrap during the morph.
    veilTitle.style.transform = 'translate(' + titDx + 'px, ' + titDy + 'px)';

    // Once the morph is complete, swap to `intro-done`: the veil fades out and
    // the in-flow content (already at the same positions) becomes visible.
    setTimeout(function () {
      stopIntroVideos();
      html.classList.remove('intro-fading');
      html.classList.add('intro-done');
    }, MORPH_MS);

    setTimeout(cleanup, MORPH_MS + FADE_MS);
  }

  function cleanup() {
    var veil = document.getElementById('intro-veil');
    if (veil) veil.remove();
    window.removeEventListener('resize', syncIntroTitleMetrics);
    html.classList.remove('intro-done');
  }

  function syncIntroTitleMetrics() {
    var veilTitle = document.querySelector('#intro-veil .intro-title');
    var flowTitle = document.querySelector('.paper-title');
    if (!veilTitle || !flowTitle) return;

    var rect = flowTitle.getBoundingClientRect();
    var style = getComputedStyle(flowTitle);
    if (rect.width > 0) {
      veilTitle.style.width = rect.width + 'px';
    }
    veilTitle.style.maxWidth = style.maxWidth;
    veilTitle.style.fontFamily = style.fontFamily;
    veilTitle.style.fontSize = style.fontSize;
    veilTitle.style.fontWeight = style.fontWeight;
    veilTitle.style.lineHeight = style.lineHeight;
    veilTitle.style.letterSpacing = style.letterSpacing;
    veilTitle.style.textWrap = style.textWrap;
    veilTitle.style.overflowWrap = style.overflowWrap;
  }

  function playIntroVideos() {
    var videos = document.querySelectorAll('#intro-veil video');
    Array.prototype.forEach.call(videos, function (video) {
      video.muted = true;
      video.defaultMuted = true;
      video.playsInline = true;
      var playAttempt = video.play();
      if (playAttempt && typeof playAttempt.catch === 'function') {
        playAttempt.catch(function () {
          video.addEventListener('canplay', function () {
            video.play().catch(function () {});
          }, { once: true });
        });
      }
    });
  }

  function stopIntroVideos() {
    var videos = document.querySelectorAll('#intro-veil video');
    Array.prototype.forEach.call(videos, function (video) {
      video.pause();
      video.style.visibility = 'hidden';
    });
  }

  // Trigger handlers — first scroll/touch/keypress dismisses the intro.
  var opts = { passive: true, once: true };
  window.addEventListener('wheel',     exit, opts);
  window.addEventListener('touchmove', exit, opts);
  var veilEl = document.getElementById('intro-veil');
  if (veilEl) {
    veilEl.addEventListener('click', exit, opts);
    veilEl.addEventListener('pointerdown', playIntroVideos, { passive: true, once: true });
  }

  function onKey(e) {
    var keys = ['ArrowDown', 'PageDown', 'End', ' ', 'Spacebar', 'Enter'];
    if (keys.indexOf(e.key) !== -1) {
      window.removeEventListener('keydown', onKey);
      exit();
    }
  }
  window.addEventListener('keydown', onKey);
})();

// Copy BibTeX to clipboard.
const copyBtn = document.getElementById('copy-bibtex');
if (copyBtn) {
  copyBtn.addEventListener('click', async () => {
    const code = document.querySelector('#bibtex pre code');
    if (!code) return;

    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      const label = copyBtn.querySelector('span');
      const original = label.textContent;
      label.textContent = 'Copied!';
      setTimeout(() => (label.textContent = original), 1500);
    } catch (_) {
      // Fallback: select the text so the user can copy manually.
      const range = document.createRange();
      range.selectNodeContents(code);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
}
