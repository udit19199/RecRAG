Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 2 items.

=== ITEM id=ba001fc0d9a45cec ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve made big improvements with its new iteration.
  Fact 0.1: Valve has made a gaggle of minor physical upgrades with its new iteration.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has Steam Deck OLED units ready to ship on November 16th.

Retrieved items:
<item 1>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.

Graph facts:
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Switch OLED
- The Steam Deck OLED improves much more than its screen -MENTIONS-> battery
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Nintendo
- The Steam Deck OLED improves much more than its screen -MENTIONS-> OLED screen
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Steam Deck 2

Entities: Steam Deck team, The Steam Deck OLED improves much more than its screen, Valve, Polygon Recommends, Polygon, OLED TV, Steam Deck, Steam Deck OLED
</item 1>

<item 2>
Graph query row:
polygon_article: The Steam Deck OLED improves much more than its screen
polygon_publication_date: 2023-11-09T18:00:00+00:00
polygon_publisher: Polygon
polygon_mentions: [{'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'battery'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'Nintendo'}, {'labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'name': 'OLED screen'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'Steam Deck team'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Switch OLED'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'],
'name': 'Steam Deck 2'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'OLED'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'HDR
OLED'}, {'labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'name': 'BIOS update'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'iFixit'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'screens'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'power
supply'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'carrying case'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'slim
case'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'original bulky case'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'Valve'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': '1 TB OLED'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name':
'512 GB OLED'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': '256 GB non-OLED Steam Deck'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'special edition 1 TB Steam Deck OLED'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'screen'}, {'labels':
['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Steam storefront'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Bluetooth'}, {'labels':
['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Polygon Recommends'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'OLED TV'}, {'labels':
['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Canada'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'batteries'}, {'labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Vox Media'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Steam Deck'}, {'labels':
['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Steam Deck OLED'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'battery life'}, {'labels':
['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Wi-Fi 6E'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'U.S.'}, {'labels':
['__KGBuilder__', '__Entity__', 'Thing'], 'name': 'Polygon Recommends badge'}, {'labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'name': 'What to Play'},
{'labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'name': 'What to Watch'}, {'labels': ['__KGBuilder__', '__Entity__', 'Publisher'], 'name': 'The Verge'}]
polygon_passages: ['The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)\nIn early 2022, Valve released my favorite video
game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of my sprawling collection of video games wherever
I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much better games looked on our family’s OLED TV.\n\nAnd
yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.\n\nPolygon Recommends is our way of
endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.', 'The Steam Deck OLED improves much more than its screen
(Polygon, 2023-11-09T18:00:00+00:00)\nWhen we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking,
entertaining, inventive, or fun — and worth fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to
Watch .\n\nIt appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but
not its guts. In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to
provide developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three
years) will have a generational power leap.\n\nFor those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t
upped the horsepower, it has improved practically everything else.', 'The Steam Deck OLED improves much more than its screen (Polygon,
2023-11-09T18:00:00+00:00)\nBecause while Valve hasn’t upped the horsepower, it has improved practically everything else. The Steam Deck OLED has longer battery
life, faster Wi-Fi, and a stand-alone Bluetooth module. And the screen isn’t just OLED; it’s HDR OLED — the first HDR OLED screen on any gaming portable, with a
peak brightness of 1,000 nits for HDR content and 600 nits for SDR (for comparison, the original screen’s peak brightness was 400 nits). And somehow, with all
of these additions, the Steam Deck OLED is lighter.\n\nSo, goodbye, old Steam Deck; hello, new Steam Deck.\n\nThe Steam Deck OLED delivers on Valve’s original
promise\n\nOLED is in the name for a reason.\n\nThe OLED panel in the new Steam Deck is lighter, thinner, and more power efficient than its LCD predecessor.
It’s also bigger and faster. Though the device has the same shape as the original Steam Deck, smaller screen bezels allow for a display that’s slightly larger,
at 7.4 inches compared to 7 inches.', 'The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)\nThe panel has a wider color
gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.\n\nSteam Deck OLED’s bigger, better battery
Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck running its original firmware (which had
poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck owner, expect those big battery
improvements over your current Steam Deck experience.\n\nThe OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In
case that jargon means nothing to you, suffice it to say that gameplay on the Steam Deck OLED looks smoother.', 'The Steam Deck OLED improves much more than its
screen (Polygon, 2023-11-09T18:00:00+00:00)\nThe OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to
carve out more of it) that lets Valve’s new handheld truly deliver on its promise of a portable gaming PC.\n\nAll the tiny, delightful improvements of Steam
Deck OLED\n\nSince the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will
continue with both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s
upgrades. The original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.\n\nSome changes — even smallish ones —
demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.', 'The Steam Deck OLED
improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)\nThe Steam Deck OLED now includes a Wi-Fi 6E radio (great if you have a modern router
and connection to make use of it) and a dedicated Bluetooth module, which allows for improved audio quality with wireless headphones, the option to wake the
Steam Deck from a controller, and the capability to connect many Bluetooth controllers for multiplayer. The haptics are noticeably improved, with a wider
frequency range. And the touchscreen is markedly better — it’s responsive to the point where I don’t feel the need to plug in a USB-C keyboard anytime I need to
type a few words.\n\nA review for spec specifics If you’re the sort of person who likes the nitty-gritty technical details of screens and batteries, we also
recommend reading The Verge’s review.\n\nValve also claims that the Steam Deck OLED is more repairable, which, mercifully, I didn’t have the need to test for
myself. (Parts will once again be sold through iFixit.)', 'The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)\n(Parts
will once again be sold through iFixit.) However, I did enjoy the new power supply, with its faster charging (45 minutes gets you from 20% to 80%) and 2.5-meter
cable that stretches the length of my couch.\n\nOne small, charming detail: The carrying case now has a tinier carrying case within it. If you want to be extra
cautious, stick to the original bulky case. But if you need to toss the Steam Deck in your backpack for a day trip? Go with the slim case that has funny little
thumbstick nipples!\n\nAll of the tweaks, large and small, amount to a perfect handheld — made better by the price, which remains the same as the original Steam
Deck available today, just that the OLED will come with more stuff.', 'The Steam Deck OLED improves much more than its screen (Polygon,
2023-11-09T18:00:00+00:00)\nValve will be selling three models:\n\n$649 - 1 TB OLED (up from 512 GB)\n\n$529 - 512 GB OLED (up from 256 GB)\n\n$399 - 256 GB
non-OLED Steam Deck (up from 64 GB)\n\nPlus, for a limited time, the company will offer a special edition 1 TB Steam Deck OLED made with a translucent shell for
$679 — only in the U.S. and Canada. Though let’s be real: This thing will sell like hotcakes and Valve will reconsider the definition of “limited supply.” Have
I considered ordering one of these limited editions? Yes. But I am also practicing restraint. Like, how many Steam Decks should one human have?\n\nI suppose
that’s the question, really. If you already have a Steam Deck, is the OLED model technically an incremental improvement? Yes. But you might find that the screen
and the battery justify the double dip.', 'The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)\nYes. But you might find
that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year, the upgrade is
inescapable.\n\nAnd if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.\n\nThe Steam Deck
OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has affiliate
partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .']
engadget_article: Steam Deck OLED review: It's just better
engadget_publication_date: 2023-11-09T18:00:38+00:00
engadget_publisher: Engadget
engadget_mentions: [{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Sony'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name':
'HDR'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Switch OLED upgrade'}, {'labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'name':
'OLED screen'}, {'labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'name': 'LCD'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'OLED
screen'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'HDR OLED display'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'P3
color gamut'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Variable refresh rates (VRR)'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'custom AMD APU'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'PS4 Slim'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'PS5 Slim'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'original Steam Deck'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Product'], 'name': 'Elden Ring'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'heat management'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Product'], 'name': 'Vita'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'SteamOS update'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'name': 'VRR'}, {'labels': ['__KGBuilder__', '__Entity__', 'Event'], 'name': 'Steam Deck OLED sale'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Product'], 'name': '64GB Steam Deck LCD model'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': '512GB Steam Deck LCD model'},
{'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': '256GB Steam Deck LCD model'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name':
'512GB Steam Deck OLED model'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': '1TB Steam Deck OLED model'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Product'], 'name': 'Limited edition 1TB OLED version'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'developers at
Valve'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Ayaneo'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'Ayn'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Retroid'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name':
'Playdate'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'handheld market'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name':
'handheld PCs'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'high-fidelity mobile play'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'name': 'portable PC gaming'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Steam'}, {'labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'name': 'AMD'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'MacBook Pro'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'Steam Deck 2'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Valve'}, {'labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'name': 'Razer'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Bluetooth'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'Steam Deck'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Asus'}, {'labels': ['__KGBuilder__', '__Entity__',
'Topic'], 'name': 'Wi-Fi'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'Steam Deck OLED'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'Hades'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'battery life'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'SteamOS'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Logitech'}, {'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'name': 'Wii U'}]
engadget_passages: ["Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)\nThe first thing to know about the Steam Deck OLED is that
it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the Switch OLED upgrade, but there’s a lot more going on internally
here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics are better, and its chip and thermals are better. One thing that
hasn’t changed is its form factor — this is still one of the biggest babies on the handheld block.\n\nThe new Steam Deck’s main upgrade is its OLED screen,
which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The OLED also has smaller bezels, making the display larger
at 7.4 inches, compared with the original 7-inch screen.", "Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)\nValve Valve Steam
Deck OLED SCORE 89 Engadget 89 Critics - Not yet scored N/A Users - Not yet scored N/A Pros Improved battery life\n\nBright and smooth visuals\n\nBetter
thermals\n\nFeels less top-heavy\n\nLong charging cable Cons Still ridiculously large\n\nSeriously it’s so big\n\nValve Valve Steam Deck OLED This isn't the
Steam Deck 2 some are waiting for, it's just a welcome improvement on an already good device. $549 at Steam\n\nOLED screen\n\nThe HDR OLED display looks
fantastic. It features a wider P3 color gamut, pure blacks and a peak brightness of 1,000 nits, compared with the original max of 400 nits. In action, this
means the screen’s bezels meld with the black of any game, creating a more polished frame for retro aspect ratios and providing room for higher-contrast colors.
The screen on the new model doesn’t support variable refresh rates (VRR), but it now tops out at 90Hz, while the original could only hit 60Hz.", "Steam Deck
OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)\nThe updated display is smooth, crisp and bright, like a perfect lemon tart. (Even then, I
do not recommend licking the screen.)\n\nBattery life\n\nOne of the downsides of the first Steam Deck was its short battery life, which maxed out at eight
hours, and that was only under ideal conditions. Valve says the new Steam Deck battery will last 30 to 50 percent longer, depending on how you’re playing, and
it charges faster — the company claims it will go from 20 percent to 80 percent in 45 minutes.\n\nCustom AMD APU\n\nThe Steam Deck OLED has a die-shrunk version
of the custom AMD APU found in the original Steam Deck — same CPU and GPU, just more efficient. This is the process Sony takes advantage of to build the PS4
Slim and PS5 Slim, but instead of collapsing the Steam Deck around smaller hardware, Valve chose to add a bigger battery and larger fan to its existing
chassis.", "Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)\nFor developers, this means the device has the same performance
targets as the original and there’s no need to create games specifically for the new hardware. In the end, the device weighs about 30 grams less than the
original model, or the equivalent of five quarters.\n\nEngadget\n\nIn use\n\nWhich leads us to the bad news for my tiny hand gang. The Steam Deck OLED is the
same monstrous size as the original: 11.7 inches long, 4.5 inches tall and nearly 2 inches thick. The thumbsticks on the new model are actually 1mm taller than
the first version. This was one of my main complaints about the Steam Deck when I first reviewed it: I have smaller-than-average hands, and the Steam Deck looks
and feels absolutely ridiculous when I’m holding it. It’s just massive, no matter how long my manicure gets. However, the OLED model feels more balanced than
the original. It’s less top-heavy, and it does technically weigh less.", "Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)\nIt’s
less top-heavy, and it does technically weigh less. These small adjustments have made a noticeable difference during my playtime, and the device feels slightly
more manageable as a handheld — or maybe I’m just too distracted by all of its bright, pretty colors to care about the cramping in my palms.\n\nThe thumbsticks
on the new handheld have an improved texture that shouldn’t turn gray after a few months’ use, and they feel perfectly serviceable. In my hands-on time so far,
I've found that the OLED touchscreen responds immediately to input, and the haptic feedback on the built-in trackpads feels even more precise, populating in
reactive pin-pricks under the textured plastic.\n\nI spent a few cozy nights on my couch swapping between the LCD and OLED Steam Decks, and honestly, after I’m
done with this review, I’ll probably retire my original model. The Steam Deck OLED is sharper and more responsive; indie games, AAA games and plain old Steam
menus look so much better on the new hardware.", "Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)\nEngadget\n\nHades is my
most-played game on the original Steam Deck, so I used that to test out Valve’s battery life claims. Though it doesn’t support HDR, a game like Hades pops on
the OLED, and its animations look more fluid than on the LCD version. The OLED lasted four hours and 23 minutes before dying. With the same settings (800p with
a 60 fps cap and similar brightness) the original model died after two hours and 42 minutes. That’s a 62 percent improvement, surpassing Valve’s own estimates,
though batteries degrade and I’ve had my LCD Steam Deck since the device’s launch in early 2022. In a stress test that involved playing Elden Ring with
everything maxed out, the OLED battery lasted two hours and 20 minutes, about 40 percent longer than the original.\n\nPlaying the Steam Decks back-to-back
highlighted another significant quality-of-life improvement: heat management.", "Steam Deck OLED review: It's just better (Engadget,
2023-11-09T18:00:38+00:00)\nPlaying the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management. My palms have
gotten toasty playing the OLED model, but they haven’t broken a sweat like they often do with the original. The new fan may be larger, but it’s not any louder,
and it clearly works a treat.\n\nThe Steam Deck OLED has a new Wi-Fi 6E module and improved antennas that should enable faster downloads for anyone with a
compatible router. I don’t have a 6E router or multi-gigabit Wi-Fi, but with a regular Wi-Fi 6 router, the top download speed I saw on the new Steam Deck was
562mbps, which was about 10 percent faster than the old Steam Deck, and 10 percent slower than the speeds I get on my MacBook Pro. The device uses Bluetooth 5.3
and adds a dedicated antenna, which should improve functionality when using Wi-Fi and Bluetooth at the same time.", "Steam Deck OLED review: It's just better
(Engadget, 2023-11-09T18:00:38+00:00)\nEngadget\n\nEven with improvements to the battery, thermals and weight distribution, the Steam Deck OLED is still more of
a Wii U than a Vita. It’s a beast of a handheld, and it makes the most sense as a portable console to use around the house, rather than a truly travel-ready
device. The new model even comes with an extra-long, 2.5m charging cable, designed specifically with couch-based players in mind. I didn’t test out the docking
station with the new Steam Deck, but Valve recently released a SteamOS update that lets players take advantage of HDR and VRR on external displays, which only
drives home the Wii U-ness of it all.\n\nAs far as software, you know the drill by now. Valve has rolled out labels for Steam games signifying how well they’ll
work on the handheld, with four levels: unknown, unsupported, playable and verified. Playable games generally come with a small caveat, like having to deploy
the on-screen keyboard, while verified games have been fully optimized for portable play.", "Steam Deck OLED review: It's just better (Engadget,
2023-11-09T18:00:38+00:00)\nCurrently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.\n\nWhen will the
Steam Deck OLED be available?\n\nThe Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company
has lowered the price of its LCD lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB
LCD model at a reduced price of $400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED
version priced at $680 that comes with a special carrying case, a translucent gray body and orange accents.\n\nValve\n\nWrap-up\n\nAccording to the developers I
spoke with at Valve, this is the definitive version of the Steam Deck.", "Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)\nThe
HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly — in
just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to build
hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies like
Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).\n\nSo, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the
facetuned, photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.", "Steam Deck OLED review: It's just better
(Engadget, 2023-11-09T18:00:38+00:00)\nThe updated display is the highlight of the device, while things like faster charging, improved antennas and smoother
haptics are welcome bonuses. More importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the
granddaddy of handheld PCs and Valve has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2,
which may be the final version of the device — unless Valve finally learns how to count to three."]
</item 2>

<item 3>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Valve Valve Steam Deck OLED SCORE 89 Engadget 89 Critics - Not yet scored N/A Users - Not yet scored N/A Pros Improved battery life

Bright and smooth visuals

Better thermals

Feels less top-heavy

Long charging cable Cons Still ridiculously large

Seriously it’s so big

Valve Valve Steam Deck OLED This isn't the Steam Deck 2 some are waiting for, it's just a welcome improvement on an already good device. $549 at Steam

OLED screen

The HDR OLED display looks fantastic. It features a wider P3 color gamut, pure blacks and a peak brightness of 1,000 nits, compared with the original max of 400
nits. In action, this means the screen’s bezels meld with the black of any game, creating a more polished frame for retro aspect ratios and providing room for
higher-contrast colors. The screen on the new model doesn’t support variable refresh rates (VRR), but it now tops out at 90Hz, while the original could only hit
60Hz.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Switch OLED upgrade
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> HDR
- Steam Deck OLED review: It's just better -MENTIONS-> LCD

Entities: OLED screen, HDR OLED display, P3 color gamut, Variable refresh rates (VRR), Steam Deck OLED review: It's just better, Steam, Steam Deck 2, Valve,
Steam Deck, Steam Deck OLED, Engadget
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.

Graph facts:
- The best video games of the year so far -MENTIONS-> Nintendo
- Microsoft Activision: What does deal mean for gamers? -MENTIONS-> Nintendo
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Nintendo
- The best October Prime Day gaming deals -MENTIONS-> Nintendo
- Super Mario Bros. Wonder review: The joy of pure imagination -MENTIONS-> Nintendo

Entities: Nintendo, Switch OLED, Steam Deck 2, The Steam Deck OLED improves much more than its screen, Valve, Polygon, Steam Deck, Steam Deck OLED, Polygon
Recommends badge, What to Play, What to Watch
</item 4>

<item 5>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> HDR
- Steam Deck OLED review: It's just better -MENTIONS-> Switch OLED upgrade
- Steam Deck OLED review: It's just better -MENTIONS-> LCD

Entities: HDR OLED display, Steam Deck OLED review: It's just better, Ayaneo, Ayn, Retroid, Playdate, handheld market, handheld PCs, high-fidelity mobile play,
Steam Deck 2, Valve, Razer, Steam Deck, Asus, Steam Deck OLED
</item 5>

<item 6>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
It’s less top-heavy, and it does technically weigh less. These small adjustments have made a noticeable difference during my playtime, and the device feels
slightly more manageable as a handheld — or maybe I’m just too distracted by all of its bright, pretty colors to care about the cramping in my palms.

The thumbsticks on the new handheld have an improved texture that shouldn’t turn gray after a few months’ use, and they feel perfectly serviceable. In my
hands-on time so far, I've found that the OLED touchscreen responds immediately to input, and the haptic feedback on the built-in trackpads feels even more
precise, populating in reactive pin-pricks under the textured plastic.

I spent a few cozy nights on my couch swapping between the LCD and OLED Steam Decks, and honestly, after I’m done with this review, I’ll probably retire my
original model. The Steam Deck OLED is sharper and more responsive; indie games, AAA games and plain old Steam menus look so much better on the new hardware.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> HDR
- Steam Deck OLED review: It's just better -MENTIONS-> Switch OLED upgrade
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> LCD

Entities: Steam Deck OLED review: It's just better, Steam, Steam Deck, Steam Deck OLED, Engadget
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck, Steam Deck OLED, Greg Coomer, The Steam Deck OLED arrives November 16 with an improved screen and longer battery life, Engadget
</item 7>

<item 8>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> HDR
- Steam Deck OLED review: It's just better -MENTIONS-> Switch OLED upgrade
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> LCD

Entities: Steam Deck OLED review: It's just better, handheld PCs, portable PC gaming, Steam Deck 2, Valve, Steam Deck, Steam Deck OLED, Engadget
</item 8>

<item 9>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.

Graph facts:
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Nintendo
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Steam Deck team
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- The Steam Deck OLED improves much more than its screen -MENTIONS-> battery
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Switch OLED

Entities: OLED screen, BIOS update, The Steam Deck OLED improves much more than its screen, Valve, Polygon, Steam Deck, Steam Deck OLED, battery life
</item 9>

<item 10>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- These Are Our Favorite Walmart Cyber Monday Deals -MENTIONS-> HDR
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> HDR
- The Steam Deck OLED improves much more than its screen -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen

Entities: HDR, Switch OLED upgrade, OLED screen, LCD, Steam Deck OLED review: It's just better, Steam Deck 2, Valve, Steam Deck, Steam Deck OLED, Engadget
</item 10>
=== END ITEM id=ba001fc0d9a45cec ===

=== ITEM id=b8910cf121bef5d7 ===
Gold sentences and facts:
Sentence 0: Doe's story was adapted into the 1971 book "The Exorcist" by William Peter Blatty, which was then made into the 1973 movie starring Ellen Burstyn and Linda Blair.
  Fact 0.0: Doe's story was adapted into the 1971 book The Exorcist by William Peter Blatty.
  Fact 0.1: The 1971 book The Exorcist was made into the 1973 movie.
  Fact 0.2: The 1973 movie starred Ellen Burstyn.
  Fact 0.3: The 1973 movie starred Linda Blair.
Sentence 1: Clive Barker’s 1987 directorial debut adapts his 1986 novella The Hellbound Heart to tell the story of Larry (Andrew Robinson) and Julia Cotton (Clare Higgins).
  Fact 1.0: Clive Barker's 1987 directorial debut adapts his 1986 novella The Hellbound Heart.
  Fact 1.1: Clive Barker's 1987 directorial debut tells the story of Larry, played by Andrew Robinson.
  Fact 1.2: Clive Barker's 1987 directorial debut tells the story of Julia Cotton, played by Clare Higgins.

Retrieved items:
<item 1>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The story of The Exorcist III’s production is a tangled mess of false starts, studio interference, and thwarted artistic ambitions. Though it was originally
conceived with Friedkin in mind to return as director, he eventually left the project due to creative differences with Blatty. John Carpenter was at one time
tapped to direct, before he too stepped aside to allow Blatty himself to take the reins. Morgan Creek Productions, which financed the film, demanded Blatty
reshoot the film’s final act, which originally ended without an exorcism. Even given these complications, The Exorcist III remains a fascinating and terrifying
film in its own right, with a hodgepodge of fascinating creative choices and literary dialogue that makes the film feel like a sister piece to Blatty’s The
Ninth Configuration. It may not have been the Exorcist sequel that audiences at the time wanted, but it’s still a damn fine picture. —TE

Graph facts:
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Amazon
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Tubi
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Apple
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Murphy

Entities: The Halloween Countdown: 31 days of horror to watch, The Exorcist III, Friedkin, Blatty, Morgan Creek Productions, The Ninth Configuration, Polygon,
John Carpenter
</item 1>

<item 2>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
After inadvertently being resurrected by a drop of blood spilled by Larry on the floor of the house’s attic, Frank seduces Julia into luring new men to the
house so that he can drain their life force and fully regain his mortal form. Surrounding this core narrative is the the story of the Lament Configuration, a
puzzle box Frank acquired before his untimely death. When solved, it conjures hellish beings known as Cenobites to the mortal plane of existence, which indulge
in hellish exercises of sadomasochistic mutilation. Easily the best and most enduring of the Hellraiser movie series, Barker’s 1987 original is a must-watch for
horror fans. —TE

Hereditary

Director: Ari Aster

Cast: Toni Collette, Alex Wolff, Milly Shapiro

Where to watch: Max

Hereditary is a victim of its own success.
</item 2>

<item 3>
Graph query row:
article: The best horror movies you can watch right now
published_on: 2023-09-28T15:26:41+00:00
passage: The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
Your time is precious, and your options are endless. The good news: We watch it all so you don’t have to. Polygon’s What to Watch highlights the best, the
funniest, the scariest, and the most exciting in movies, TV, and everything in between. Stop scrolling, start watching!

Whether it’s something gory and macabre, silly and irreverent, or eerie and unsettling, the genre of horror is as rich and varied as the multitude of ghosts,
ghoulies, and homicidal maniacs that go bump in the night.

Looking for the best horror films available to stream on Netflix, Hulu, Max, and Paramount Plus? No worries, we’ve got the goods. We’ve combed through the
libraries of each of the major streaming platforms to bring you a list of our most recommended horror movies. Here are the best horror movies you can stream
right now, from old classics to new hits. Our latest update added Encounters of the Spooky Kind.
</item 3>

<item 4>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The Texas Chainsaw Massacre 2 is nothing like that. You won’t see its name in a film school syllabus, and the films it inspired can be counted on one hand.
Nonetheless, there’s something commendable about the gall of Tobe Hooper, the original Texas Chain Saw director, returning to his series with the energy of a
middle-aged man on his first ayahuasca retreat.

Leatherface gets in a high-speed car battle, eventually standing on the roof of a speeding sedan. Dennis Hopper dual-wields chainsaws. A radio station office is
obliterated like it’s a novelty rage room. Sometimes you want a steak, and sometimes you want a hamburger. Texas Chainsaw Massacre 2 is a damn juicy burger. —CP

Oct. 27: Possessor

Where to watch: Hulu

My first Brandon Cronenberg (yes, David’s son) movie was Infinity Pool, his latest messed-up meditation on the rich and powerful.

Graph facts:
- Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg -MENTIONS-> David
- Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg -MENTIONS-> David
- 8 authors and their 8 gift book picks for this holiday season -MENTIONS-> David
- NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable -MENTIONS-> David
- The best comedies you’ll find on streaming right now -MENTIONS-> David

Entities: David, The Halloween Countdown: 31 days of horror to watch, The Texas Chainsaw Massacre 2, Texas Chain Saw, Leatherface, Dennis Hopper, Possessor,
Brandon Cronenberg, Infinity Pool, Polygon, Hulu, horror, Tobe Hooper
</item 4>

<item 5>
The best horror movies you can watch right now (Polygon, 2023-09-28T15:26:41+00:00)
An unsettling tale about a plastic surgeon (played by Pierre Brasseur) who kidnaps young women and performs surgery on them to try and find a face replacement
for his daughter (Édith Scob), Eyes Without a Face is equal parts haunting and beautiful. Scob’s iconic face mask in the movie was later referenced in her role
in the also-excellent Holy Motors many decades later. —PV

Hellraiser

Director: Clive Barker

Cast: Andrew Robinson, Clare Higgins, Ashley Laurence

Where to watch: Prime Video, AMC+, Shudder, Tubi, Pluto, Hoopla

Clive Barker’s 1987 directorial debut adapts his 1986 novella The Hellbound Heart to tell the story of Larry (Andrew Robinson) and Julia Cotton (Clare Higgins).
The Cottons are a married couple who move into the home of Larry’s recently deceased brother, Frank (Sean Chapman), with whom Julia had a previous affair.

Graph facts:
- 9 great movies to watch on Tubi for free right now -MENTIONS-> Tubi
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Tubi
- Watch great noir this Noirvember -MENTIONS-> Tubi
- The best movies of 2023 so far -MENTIONS-> Tubi
- The best comedies you’ll find on streaming right now -MENTIONS-> Tubi

Entities: Tubi, Polygon, Pierre Brasseur, Édith Scob, The best horror movies you can watch right now, Hellraiser, Clive Barker, The Hellbound Heart, Eyes
Without a Face, Holy Motors, Andrew Robinson, Clare Higgins, Ashley Laurence, Prime Video, AMC+
</item 5>

<item 6>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
—PV

Oct. 31: The Exorcist III

Where to watch: Prime Video, Criterion Channel

Nearly 50 years since it was first released in theaters, William Friedkin’s The Exorcist has withstood the test of time as one of the most terrifying and
influential horror films in American cinema. The sequel that came closest to capturing that same lightning in a bottle is The Exorcist III, written and directed
by William Peter Blatty, the original screenwriter of the first film and author of the novel on which it was based.

When a series of gruesome murders matching the modus operandi of an infamous serial killer begins to terrorize the city of Georgetown, Lt. William F. Kinderman
(George C. Scott) investigates the psych ward of a local hospital in search of leads. Confronted with a suspect whose appearance matches that of his late friend
Father Damien Karras, Kinderman finds himself thrust into a battle for his very soul as he confronts an evil beyond his comprehension.

Graph facts:
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Amazon
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Tubi
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Apple
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Murphy

Entities: The Halloween Countdown: 31 days of horror to watch, The Exorcist III, William Friedkin, William F. Kinderman, George C. Scott, Father Damien Karras,
Georgetown, horror films, American cinema, Polygon, The Exorcist, William Peter Blatty, Criterion Channel, Prime Video
</item 6>

<item 7>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Every person reads like a threat, and every new competitor or friend seems to be looking for a new way to drain the life out of the people around them — until
they finally actually do it. But, like all Refn projects, what really sets The Neon Demon apart is how it looks. The griminess of the movie’s world turns
brightly lit photo shoots into shadow-filled nightmares, with people who stalk the sets like jungle predators.

Though The Neon Demon didn’t get the same acclaim as other so-called elevated horror movies from the time, looking back, it fits perfectly among the ranks of
The Witch, It Follows, and Under the Skin. Refn’s nightmarish modeling movie is dripping with cynicism, cruelty, and a creeping atmosphere, with more meanness
than almost any other movie from the era could muster. But it’s all in the pursuit of beauty, of course. —AG

Oct. 16: Project Wolf Hunting

Where to watch: Hi-Yah!

Graph facts:
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Tubi
- The best movies of 2023 so far -MENTIONS-> Hi-Yah!
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Amazon
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Apple

Entities: Hi-Yah!, The Neon Demon, The Witch, Under the Skin, Refn, Project Wolf Hunting, The Halloween Countdown: 31 days of horror to watch, It Follows,
Polygon
</item 7>

<item 8>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Enjoyment of Dracula 2000 may vary depending on a tolerance for Hot Topic goofs, but if you don’t catch up with it, you can’t enjoy Lussier’s sequel, Dracula
II: Ascension, starring Jason Scott Lee as a martial-arts-fighting priest who kicks Dracula’s ass! —Matt Patches

Oct. 7: Prince of Darkness

Where to watch: Peacock

There aren’t enough movies about the end of the world. Sure, there are plenty of movies about preventing catastrophe at that scale, but rarely do they have the
follow-through to truly get existentially terrifying. Thankfully, John Carpenter is no coward, and he made three movies about different kinds of apocalypses.
But while The Thing is the best known, and In the Mouth of Madness is the zaniest, Prince of Darkness might be the scariest of the trilogy.

Prince of Darkness follows a group of college researchers who are sent to study a strange canister found in the basement of a church.

Graph facts:
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Tubi
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Apple
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Amazon
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Murphy

Entities: Dracula 2000, Lussier, Dracula II: Ascension, Jason Scott Lee, Prince of Darkness, The Thing, In the Mouth of Madness, college researchers, canister,
church, end of the world, apocalypses, The Halloween Countdown: 31 days of horror to watch, Polygon, Dracula
</item 8>

<item 9>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
Even still, Halloween is an especially spooky time of year, and it warrants special attention and celebration.

For the past three years, Polygon has put together a Halloween countdown calendar, selecting 31 of our staff’s top horror-themed or Halloween-adjacent picks
across movies and TV throughout the month of October, all available to watch at home. We’ve loved doing it, so much so that we’re bringing it back again — this
time with a whole new batch of films and shows to choose from.

Every day for the month of October, we’ll add a new recommendation to this countdown and tell you where you can watch it. So curl up on the couch, dim the
lights, and grab some popcorn for a terrifying and entertaining marathon of horrific delights.

Oct. 1: Messiah of Evil

Where to watch: Prime Video, Shudder, Pluto TV, Plex

A undersung classic of ’70s cosmic horror, Willard Huyck and Gloria Katz’s Messiah of Evil is a surreal and nightmarish experience that more than deserves its
status as a cult horror classic.

Graph facts:
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Tubi
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Apple
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Amazon
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Murphy

Entities: Messiah of Evil, Willard Huyck, Gloria Katz, The Halloween Countdown: 31 days of horror to watch, Pluto TV, Plex, Halloween, Polygon, horror, Prime
Video, Shudder
</item 9>

<item 10>
The Halloween Countdown: 31 days of horror to watch (Polygon, 2023-10-10T15:27:53+00:00)
The film centers on Arletty, a young woman who journeys to the remote coastal town of Point Dume, California, to track down her missing father, a reclusive
painter who is plagued by disturbing visions of an impending apocalypse. After crossing paths with an eccentric aristocrat and his groupie companions, the group
bears witness to the realization of the painter’s prophecy, as the townspeople are transformed into flesh-eating ghouls and the messiah of evil makes his return
to the mortal realm.

At times convoluted and bizarre, Messiah of Evil is a genuinely entertaining horror thriller filled with memorable scares and chilling set-pieces. From a
cross-eyed albino man gleefully devouring a field mouse whole to a unsuspecting woman watching a Western as the theater around her slowly fills with ravenous
undead zombies, it’s a dreamy and frightening experience that lands somewhere between the ineffable gothic horror of H.P. Lovecraft and the anti-consumerist
allegories of George A. Romero.

Graph facts:
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Tubi
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Apple
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Amazon
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Murphy

Entities: Messiah of Evil, Arletty, Point Dume, H.P. Lovecraft, The Halloween Countdown: 31 days of horror to watch, Western, California, Polygon, George A.
Romero
</item 10>
=== END ITEM id=b8910cf121bef5d7 ===
