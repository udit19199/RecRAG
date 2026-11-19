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

This file has 3 items.

=== ITEM id=944f660d8502b0de ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made these minor physical upgrades alongside the big improvements.
  Fact 0.2: Valve's new iteration includes big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units of the Steam Deck OLED ready to ship on that same day.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck OLED, The Steam Deck OLED arrives November 16 with an improved screen and longer battery life, Pierre-Loup Griffais, handheld
gaming, handhelds, 1TB model, Limited Edition version, 512GB LCD model, Engadget
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .

Graph facts:
- The Steam Deck OLED improves much more than its screen -MENTIONS-> battery
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Nintendo
- The Steam Deck OLED improves much more than its screen -MENTIONS-> OLED screen
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Steam Deck team
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Switch OLED

Entities: The Steam Deck OLED improves much more than its screen, Valve, Steam storefront, Polygon, Vox Media, Steam Deck, Steam Deck OLED
</item 4>

<item 5>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck, Steam Deck OLED, Steam Deck OLED arrival, Hades, battery life, The Steam Deck OLED arrives November 16 with an improved screen and
longer battery life, Engadget
</item 5>

<item 6>
Graph query row:
publisher: Polygon
article: The Steam Deck OLED improves much more than its screen
passage: The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
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
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck, Steam Deck OLED, The Steam Deck OLED arrives November 16 with an improved screen and longer battery life, Steam Machine, Steam
Controller, Griffais, SteamOS, Coomer, Engadget
</item 8>

<item 9>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Valve
- The best October Prime Day gaming deals -MENTIONS-> Valve
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Valve
- The Steam Deck OLED improves much more than its screen -MENTIONS-> Valve
- Steam Deck OLED review: It's just better -MENTIONS-> Valve

Entities: Valve, Steam Deck OLED, Wi-Fi 6E, Bluetooth 5.3, antenna array, power supply, thumbsticks, touch pads, The Steam Deck OLED arrives November 16 with an
improved screen and longer battery life, Engadget
</item 9>

<item 10>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Switch OLED upgrade
- Steam Deck OLED review: It's just better -MENTIONS-> OLED screen
- Steam Deck OLED review: It's just better -MENTIONS-> Sony
- Steam Deck OLED review: It's just better -MENTIONS-> HDR
- Steam Deck OLED review: It's just better -MENTIONS-> LCD

Entities: Steam Deck OLED sale, 64GB Steam Deck LCD model, 512GB Steam Deck LCD model, 256GB Steam Deck LCD model, 512GB Steam Deck OLED model, 1TB Steam Deck
OLED model, Limited edition 1TB OLED version, developers at Valve, Steam Deck OLED review: It's just better, Steam, Valve, Steam Deck OLED, Engadget
</item 10>
=== END ITEM id=944f660d8502b0de ===

=== ITEM id=44f96b012780b0ec ===
Gold sentences and facts:
Sentence 0: Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in the semi-finals — and England have beaten all of them.
  Fact 0.0: Argentina has provided highlights.
  Fact 0.1: Japan has provided highlights.
  Fact 0.2: Chile has provided highlights.
  Fact 0.3: Samoa has provided highlights.
  Fact 0.4: Fiji has provided highlights.
  Fact 0.5: Los Pumas are also in the semi-finals.
  Fact 0.6: England have beaten Argentina.
  Fact 0.7: England have beaten Japan.
  Fact 0.8: England have beaten Chile.
  Fact 0.9: England have beaten Samoa.
  Fact 0.10: England have beaten Fiji.
Sentence 1: S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by Fiji and Portugal in Toulouse.
  Fact 1.0: This report is datelined Saint-Etienne.
  Fact 1.1: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com.
  Fact 1.2: The Wallabies' Rugby World Cup fate is decided by Fiji and Portugal.
  Fact 1.3: The Fiji and Portugal match is played in Toulouse.
  Fact 1.4: The Wallabies' Rugby World Cup fate is decided in Toulouse.

Retrieved items:
<item 1>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Stade de France, Saint-Denis, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream, TV channel, lineups,
odds for 2023 Rugby World Cup semifinal, England, Borthwick, the world's best, Sporting News
</item 1>

<item 2>
Graph query row:
article: Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal
publisher: Sporting News
connections: [{'entity': 'Jesse Kriel', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Siya Kolisi',
'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Cheslin Kolbe', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Pieter-Steph du Toit', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'],
'relationship': 'MENTIONS'}, {'entity': 'France', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'Handre
Pollard', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'World Cup final', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'Jacques Nienaber', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship':
'MENTIONS'}, {'entity': 'Duane Vermeulen', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Bongi Mbonambi',
'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Tom Curry', 'entity_type': ['__KGBuilder__', '__Entity__',
'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Deon Fourie', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'},
{'entity': 'Damian de Allende', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Sky Go', 'entity_type':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'beIN Sports', 'entity_type': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Paris', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'},
{'entity': 'India', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'South Africa', 'entity_type':
['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'Malaysia', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'],
'relationship': 'MENTIONS'}, {'entity': '2019 Rugby World Cup final', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'},
{'entity': 'Owen Farrell', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'South Africa', 'entity_type':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Romania', 'entity_type': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Tonga', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'entity': 'Les Bleus', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': '2023 Rugby World Cup',
'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': '2023 Rugby World Cup final', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': '2019 Rugby World Cup', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'MENTIONS'}, {'entity': 'second semi-final', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'Stade de France',
'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'Saint-Denis', 'entity_type': ['__KGBuilder__', '__Entity__',
'Place'], 'relationship': 'MENTIONS'}, {'entity': 'England vs South Africa semifinal', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'MENTIONS'}, {'entity': 'England vs South Africa 2023 Rugby World Cup semifinal', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'MENTIONS'}, {'entity': 'Steve Borthwick', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Liam Heagney',
'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'TSN', 'entity_type': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'TSN+', 'entity_type': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'},
{'entity': 'ITV X', 'entity_type': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'Channel 9', 'entity_type':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': '9Now', 'entity_type': ['__KGBuilder__', '__Entity__', 'Product'],
'relationship': 'MENTIONS'}, {'entity': 'Fan Code', 'entity_type': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity':
'RugbyPass', 'entity_type': ['__KGBuilder__', '__Entity__', 'Publisher'], 'relationship': 'MENTIONS'}, {'entity': 'Marcus Smith', 'entity_type':
['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'England Rugby', 'entity_type': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'entity': '2023 Rugby World Cup semifinal', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'],
'relationship': 'MENTIONS'}, {'entity': 'Freddie Steward', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity':
'Jonny May', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Joe Marchant', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Manu Tuilagi', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship':
'MENTIONS'}, {'entity': 'Elliot Daly', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Alex Mitchell',
'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Joe Marler', 'entity_type': ['__KGBuilder__', '__Entity__',
'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Jamie George', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'},
{'entity': 'Dan Cole', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Maro Itoje', 'entity_type':
['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'George Martin', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'],
'relationship': 'MENTIONS'}, {'entity': 'Courtney Lawes', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Ben
Earl', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Genge', 'entity_type': ['__KGBuilder__', '__Entity__',
'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Chessum', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity':
'quarter-final with France', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'Theo Dan', 'entity_type':
['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Ellis Genge', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'],
'relationship': 'MENTIONS'}, {'entity': 'Kyle Sinckler', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity':
'Ollie Chessum', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Billy Vunipola', 'entity_type':
['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Danny Care', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'],
'relationship': 'MENTIONS'}, {'entity': 'George Ford', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Ollie
Lawrence', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Kurt-Lee Arendse', 'entity_type':
['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Eben Etzebeth', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'],
'relationship': 'MENTIONS'}, {'entity': 'Damian Willemse', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity':
'Manie Libbok', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Cobus Reinach', 'entity_type':
['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Steven Kitshoff', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'],
'relationship': 'MENTIONS'}, {'entity': 'Frans Malherbe', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity':
'Franco Mostert', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Ox Nche', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Vincent Koch', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship':
'MENTIONS'}, {'entity': 'RG Snyman', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Kwagga Smith',
'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Faf de Klerk', 'entity_type': ['__KGBuilder__', '__Entity__',
'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Willie le Roux', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'},
{'entity': 'England', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Japan', 'entity_type':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Chile', 'entity_type': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Samoa', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'entity': "South Africa's 27-13 victory at Twickenham", 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity':
'Twickenham', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'Borthwick', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': "the world's best", 'entity_type': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship':
'MENTIONS'}, {'entity': 'Marseille', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'Australia',
'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': '2019 World Cup final', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'Rugby World Cup', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'entity': 'Fiji', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Rugby
World Cup', 'entity_type': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'ITV', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Scotland', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'entity': 'Eddie Jones', 'entity_type': ['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity':
'France', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Thomas Ramos', 'entity_type':
['__KGBuilder__', '__Entity__', 'Person'], 'relationship': 'MENTIONS'}, {'entity': 'Canada', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'],
'relationship': 'MENTIONS'}, {'entity': 'Stan Sport', 'entity_type': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'World
Rugby', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'All Blacks', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Singapore', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship':
'MENTIONS'}, {'entity': 'Argentina', 'entity_type': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Hong Kong',
'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'Ireland', 'entity_type': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'New Zealand', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'},
{'entity': 'UK', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'Peacock', 'entity_type': ['__KGBuilder__',
'__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'USA', 'entity_type': ['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}]
</item 2>

<item 3>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.

Graph facts:
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Owen Farrell
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa

Entities: Owen Farrell, South Africa, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023
Rugby World Cup semifinal, England, Japan, Chile, Samoa, Marseille, Sporting News, Fiji, Argentina
</item 3>

<item 4>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Steve Borthwick, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023
Rugby World Cup semifinal, England, Japan, Chile, Samoa, South Africa's 27-13 victory at Twickenham, Twickenham, 2019 World Cup final, Sporting News, Fiji,
Eddie Jones, Argentina
</item 4>

<item 5>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Paris
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> beIN Sports
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Sky Go
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Paris
- There’s something going on with AI startups in France -MENTIONS-> Paris

Entities: Sky Go, beIN Sports, Paris, India, Malaysia, South Africa, England vs South Africa 2023 Rugby World Cup semifinal, Steve Borthwick, Liam Heagney, TSN,
TSN+, ITV X, Channel 9, 9Now, Fan Code
</item 5>

<item 6>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa
- Cheslin Kolbe -PARTICIPATED_IN-> RWC final
- Cheslin Kolbe -PARTICIPATED_IN-> quarter-final with France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Cheslin Kolbe
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> South Africa

Entities: Cheslin Kolbe, South Africa, 2023 Rugby World Cup final, 2019 Rugby World Cup, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa:
Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal, England, Sporting News, Rugby World Cup, All Blacks
</item 6>

<item 7>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Steve Borthwick, RugbyPass, Marcus Smith, 2023 Rugby World Cup semifinal, Freddie Steward, Manu Tuilagi, Joe Marler, George Martin,
Ellis Genge, Ollie Chessum, Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal, England, Sporting
News, Fiji
</item 7>

<item 8>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa

Entities: South Africa, Romania, Tonga, Les Bleus, 2023 Rugby World Cup, Stade de France, 2023 Rugby World Cup semifinal, Where to watch England vs South
Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal, England, Sporting News, Scotland, All Blacks, Ireland
</item 8>

<item 9>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Jacques Nienaber
- Cheslin Kolbe -PARTICIPATED_IN-> RWC final
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Cheslin Kolbe
- Cheslin Kolbe -PARTICIPATED_IN-> Rugby World Cup
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Jacques Nienaber

Entities: Cheslin Kolbe, Jacques Nienaber, Damian de Allende, South Africa, 2023 Rugby World Cup semifinal, quarter-final with France, Theo Dan, Ellis Genge,
Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence, Kurt-Lee Arendse
</item 9>

<item 10>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'

Graph facts:
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> South Africa
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> South Africa
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> South Africa
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> South Africa
- ICC World Cup 2023: India at Cricket World Cup semi-finals so far -MENTIONS-> South Africa

Entities: 2019 Rugby World Cup final, South Africa, 2023 Rugby World Cup, 2023 Rugby World Cup semifinal, Where to watch England vs South Africa: Live stream,
TV channel, lineups, odds for 2023 Rugby World Cup semifinal, England, Sporting News
</item 10>
=== END ITEM id=44f96b012780b0ec ===

=== ITEM id=a26d221223555886 ===
Gold sentences and facts:
Sentence 0: After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week 9 as a big part of the offense with Josh Dobbs under center.
  Fact 0.0: Hockenson has had three limited practices this week.
  Fact 0.1: The writer will trust that Hockenson is ready to go.
  Fact 0.2: Hockenson continued to play through the ailment in Week 9.
  Fact 0.3: Hockenson was a big part of the offense in Week 9.
  Fact 0.4: Josh Dobbs was under center for the offense when Hockenson played in Week 9.
Sentence 1: Kincaid was a limited participant in practice on Wednesday due to a shoulder injury, but it’s been reported he’s likely to suit up on Saturday at less than 100%.
  Fact 1.0: Kincaid was a limited participant in practice on Wednesday.
  Fact 1.1: Kincaid's limited practice participation was due to a shoulder injury.
  Fact 1.2: It has been reported that Kincaid is likely to suit up on Saturday.
  Fact 1.3: It has been reported that Kincaid is likely to play on Saturday at less than 100%.

Retrieved items:
<item 1>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Joe Burrow
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Joe Burrow
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> Joe Burrow
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Joe Burrow
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Joe Burrow

Entities: Joe Burrow, Bills, Bengals, Curtis Samuel, Texans, Trenton Irwin, Tyler Boyd, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on
Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Bengals win over the Bills, Bengals host the Texans, Tee Higgins, Sam Hubbard
</item 1>

<item 2>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Saints
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Saints
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Saints

Entities: Saints, Curtis Samuel, Josh Dobbs, Kevin O'Connell, Sporting News, Justin Jefferson, Sunday's game against the Saints, Fantasy Injury Updates Week 10:
News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Kirk Cousins, T.J. Hockenson, fantasy
</item 2>

<item 3>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> ESPN
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> ESPN
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> ESPN
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> ESPN
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> ESPN

Entities: ESPN, Curtis Samuel, Eagles, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson
& Curtis Samuel, Ja'Marr Chase, Jeremy Fowler, Week 10, Jamison Crowder, Week 8's game against the Eagles, toe injury, Jahan Dotson, T.J. Hockenson
</item 3>

<item 4>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 4>

<item 5>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> New Orleans
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> New Orleans
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> New Orleans
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> New Orleans
- DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> New Orleans

Entities: New Orleans, NFL Network, Curtis Samuel, fantasy football, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on Justin Jefferson,
Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Ian Rapoport, Week 10, Boyd, Cincinnati, injury, T.J. Hockenson
</item 5>

<item 6>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> LaPorta
- LaPorta -PARTICIPATED_IN-> NFL 'Monday Night Football' game
- LaPorta -PARTICIPATED_IN-> Raiders vs. Lions game
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Taysom Hill
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Taysom Hill

Entities: LaPorta, Taysom Hill, Jaguars, Week 12, Kelce, Njoku, Hockenson, fantasy football, Week 14, fantasy points per game (FPPG), Chiefs, Sporting News,
Seahawks, tight end, PPR formats
</item 6>

<item 7>
Graph query row:
entity: Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel
relationship: PUBLISHED_BY
related: Sporting News
page: Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
passage: Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 7>

<item 8>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
In addition, K.J. Osborn (concussion) is also listed as "questionable," though his status steadily improved throughout the week from DNP on Wednesday to full
practice on Friday. If Jefferson is back, he's a must-start. If Jefferson is out but Osborn returns, both Osborn and Jordan Addison are worth using.

WEEK 10 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is Ja'Marr Chase playing Week 10?

UPDATE: ESPN's Jeremy Fowler reported late Saturday that the Bengals are "hopeful" that Chase will play against Houston, but they won't know for sure until
after a pre-game workout.

UPDATE 2: Chase is officially ACTIVE.

Chase (back) appears ready to take his status down to kickoff. According to The Athletic's Dianna Russini, the Bengals' star receiver will work out on Sunday
morning and his status will be determined afterward.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -MENTIONS-> The Athletic
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic

Entities: The Athletic, Bengals, Jordan Addison, Curtis Samuel, PPR rankings, K.J. Osborn, ESPN, Sporting News, Justin Jefferson, Fantasy Injury Updates Week
10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Jeremy Fowler, Dianna Russini, pre-game workout, Week 10
</item 8>

<item 9>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Minnesota, ESPN, Curtis Samuel, Josh Dobbs, Sporting News, Justin Jefferson, Fantasy Injury Updates Week 10: News on Justin Jefferson,
Ja'Marr Chase, T.J. Hockenson & Curtis Samuel, Ja'Marr Chase, Jeremy Fowler, Week 10, Seahawks, Arizona, Commanders, Week 9
</item 9>

<item 10>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Saints
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Saints
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Saints

Entities: Saints, hand injury, torn Achilles, CBSSports.com, Atlanta, New York, thigh injury, ankle injury, Jets, Hughes, injuries, Hollins, Nate Landman,
Olave, NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable
</item 10>
=== END ITEM id=a26d221223555886 ===
