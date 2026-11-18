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

=== ITEM id=a0981ea5182816b1 ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: A friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: The friendship bracelet seemingly expressed his interest in Swift.
Sentence 1: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 1.0: Swift has now shared that she has nothing to hide.
  Fact 1.1: Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 1.2: Swift has a new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Head coach Andy Reid was also asked about his star tight end's supposed relationship, to which he quipped, "I set them up."

Sept. 24: Kelce and Swift leave stadium together

"You know when it's time to go." - It's Time To Go

This was the first time the two were photographed together. They walked side by side, Kelce fittingly wearing an outfit with "1989" in the title and Swift
wearing a Chiefs zip-up.

Sept. 24: The two leave in a convertible

"Nothing good starts in the getaway car." - Getaway Car

Swift and Swift were spotted in a convertible, with the tight end driving, after the Chiefs' win. Kelce reportedly rented out a restaurant for a private party
with Swift and his teammates.

"Taylor Swift likes that Travis Kelce 'pursued her' and the two are having a 'great time getting to know each other,'" a source told Entertainment Tonight.

Graph facts:
- Taylor Swift -MENTIONS-> jersey sales
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Andy Reid
- FanDuel 'Prop Stars' Picks: Best bets for Travis Kelce & Taylor Swift-inspired props for Chiefs-Jets Week 4 -MENTIONS-> Andy Reid
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Andy Reid
- Taylor Swift -MENTIONS-> Instagram

Entities: Andy Reid, Taylor Swift, Travis Kelce, It's Time To Go, Getaway Car, Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance
between pop star, Chiefs TE, CBSSports.com, Chiefs, 1989, Entertainment Tonight
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."

Graph facts:
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift
- Taylor Swift -MENTIONS-> jersey sales

Entities: Taylor Swift, Arrowhead, rumored romance, Travis Kelce, Mahomes, postgame press conference, Swiftie talk, Taylor Swift-Travis Kelce timeline:
Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, game, Donna Kelce
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Kansas City
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Kansas City
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Kansas City
- Beyoncé tops box office with $21 million debut for concert film ‘Renaissance’ -MENTIONS-> Kansas City
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Kansas City

Entities: Kansas City, Taylor Swift, Travis Kelce, Mahomes, KSHB 41, Chiefs Kingdom, Speak Now, Enchanted, afterparty, Taylor Swift-Travis Kelce timeline:
Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, game
</item 5>

<item 6>
Graph query row:
article: Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE
published_on: 2023-10-12T23:21:52+00:00
publisher: CBSSports.com
connected_person: Taylor Swift
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Chicago Bears
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chicago Bears
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Chicago Bears
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> Chicago Bears
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Chicago Bears

Entities: Chicago Bears, Taylor Swift, Travis Kelce, Chiefs' Week 3 game against the Chicago Bears, nonprofit charity event, Vigilante Sh--, Fifteen, Taylor
Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, Donna Kelce
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But before he could even answer, the answer was given for him.

Sept. 14: Rumors aren't denied

"The rumors are terrible and cruel, but honey, most of them are true." - New Romantics

Of course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not confirm them, but he
also did not deny them.

Sept. 17: The references begin

"I watched it begin again." - Begin Again

The rumors began to pick up, and now there were a lot of people, commentators included, making Swift references around Kelce.

When the tight end caught a touchdown pass from Patrick Mahomes, CBS Sports' Ian Eagle said, "Kelce finds a blank space for the score."

This was only the start to many, many, Swift lyrics being referenced during NFL games.

Sept. 21: Kelce sends Swift an invite

"So inviting, I almost jump in."

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Ian Eagle, Jason, Taylor Swift, rumored romance, CBS Sports, New Romantics, Begin Again, Rumors aren't denied, The references begin, Kelce sends
Swift an invite, Travis Kelce, Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE, Patrick Mahomes,
CBSSports.com
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Jason Kelce, Taylor Swift, rumored romance, Travis Kelce, Taylor Swift-Travis Kelce timeline: Everything to know about rumored
romance between pop star, Chiefs TE, CBSSports.com, Chiefs
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.

Graph facts:
- Taylor Swift -MENTIONS-> jersey sales
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Taylor Swift

Entities: Taylor Swift, rumored romance, Travis Kelce, Belichick, Front Office Sports, Apple, Stubhub, Taylor Swift-Travis Kelce timeline: Everything to know
about rumored romance between pop star, Chiefs TE, Chiefs game, CBSSports.com, Chiefs, Instagram, New England Patriots
</item 10>
=== END ITEM id=a0981ea5182816b1 ===

=== ITEM id=d267083f2264bebd ===
Gold sentences and facts:
Sentence 0: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.0: This Diablo 4 guide gathers the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.1: This Diablo 4 guide simplifies the best Diablo 4 Sorcerer builds for season 2.
Sentence 1: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2.
  Fact 1.0: This Diablo 4 guide gathers the best Diablo 4 Barbarian builds for season 2.
  Fact 1.1: This Diablo 4 guide simplifies the best Diablo 4 Barbarian builds for season 2.

Retrieved items:
<item 1>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
As such, you’ll need to unlock new skills before enhancing existing ones:

Frenzy / Enhanced Frenzy Hammer of the Ancients / Enhanced Hammer of the Ancients / Furious Hammer of the Ancients Combat Frenzy Rallying Cry / Enhanced
Rallying Cry / Tactical Rallying Cry Hammer of the Ancients (ranks 2-3) Leap / Enhanced Leap / Power Leap Hammer of the Ancients (ranks 4-5) Death Blow /
Enhanced Death Blow / Warrior’s Death Blow / Death Blow (rank 2-5) Wrath of the Berserker / Prime Wrath of the Berserker / Supreme Wrath of the Berserker Pit
Fighter (ranks 1-3) Tempered Fury Furious Impulse (ranks 1-3) Walking Arsenal Hamstring No Mercy Expose Vulnerability (ranks 1-3) No Mercy (ranks 2-3) Swiftness
(ranks 1-3) Thick Skin (ranks 1-3) Counteroffensive (ranks 1-3) Defensive

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, Hammer of the Ancients, Frenzy, Death Blow, Rallying Cry, Leap, Wrath of the Berserker, Combat Frenzy, Pit Fighter, Tempered Fury, Furious
Impulse, Walking Arsenal, No Mercy, Expose Vulnerability, Swiftness
</item 1>

<item 2>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
The idea here is that you’re constantly swapping between your skills and weapons to give yourself massive damage buffs.

When making a build, you want to have one of these Key Passives in mind and choose your skills accordingly. Thankfully skills that share certain aspects (like
Bleed) are color-coded on the skill tree, and you can even use the in-game search function to find them all. Other Barbarian builds let you take advantage of
mechanics like Thorns or Shouts. But they still typically end with one of these Key Passives.

Looking for other Diablo 4 builds? We have explainers on the Sorcerer, Rogue, Druid, and Necromancer classes.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, Bleed, The best Barbarian builds and skills in Diablo 4, Key Passives, Thorns, Shouts, Sorcerer, Rogue, Druid, Necromancer, Diablo 4,
Polygon
</item 2>

<item 3>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
If you don’t want to end up dead more often than not, make sure to invest heavily in defensive skills while making your way toward one of the six Key Passives
we listed above.

Looking for other Diablo 4 builds? We have explainers on the Barbarian, Rogue, Druid, and Necromancer classes.

Graph facts:
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Enchantment
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Sanctuary
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Sorcerer builds
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> lightning
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> ice

Entities: The best Sorcerer builds and skills in Diablo 4, Sorcerer builds and skills, defensive skills, six Key Passives, Barbarian class, Rogue class, Druid
class, Necromancer class, Diablo 4, Polygon
</item 3>

<item 4>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Ring 2: Aspect of Limitless Rage

If you want to get into the nitty gritty of stat priorities, refer to AaronActionRPG’s original guide.

Paragon Boards for Double Swing Barbarian build

The Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:

Starter Board / Bloodfeeder Glyph Decimator / Ambidextrous Glyph Carnage / Might Glyph Flawless Technique / Ire Glyph Weapon Master / Territorial Glyph Bone
Breaker / Exploit Glyph Warbringer / Wrath Glyph

You should level your Glyphs to 15 in this specific order:

Exploit Territorial Ambidextrous Might Wrath Bloodfeeder Ire

You can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow AaronActionRPG’s path exactly.

How to make a great Barbarian build

Barbarians are Diablo 4’s only pure melee class, with the other classes having hybrid builds for multiple ranges.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, Paragon Board system, Glyphs, AaronActionRPG, The best Barbarian builds and skills in Diablo 4, Double Swing Barbarian build, Aspect of
Limitless Rage, Ring 2, Starter Board, Bloodfeeder Glyph, Decimator, Ambidextrous Glyph, Carnage, Might Glyph, Flawless Technique
</item 4>

<item 5>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Barbarians are a relatively simple class in Diablo 4: They smash stuff. However, Diablo 4 is underscored by math that’s so complex that even the simple act of
smashing stuff becomes a dizzying equation. That’s where we come in, with our suggestions of the best Barbarian builds for Diablo 4 season 2.

Barbarians are big, muscular, and powerful. They excel at using Bleed to kill their enemies over time. And their unique weapon-swapping system means they get to
have more Legendary powers at their disposal than any other class.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2. If you need even more information on item affixes or
other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, item affixes, Barbarian builds, Bleed, weapon-swapping system, Legendary powers, season 2, The best Barbarian builds and skills in Diablo
4, Diablo 4, Polygon
</item 5>

<item 6>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Recharging Aspect (Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana

(Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana Aspect of Control (Sunken Library in Kehjistan) — You deal X% more damage
to Immobilized, Stunned, or Frozen enemies

(Sunken Library in Kehjistan) — You deal X% more damage to Immobilized, Stunned, or Frozen enemies Edgemaster’s Aspect (Oldstones dungeon in Scosglen) — Skills
deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary Resource

Sorcerer Season 2 endgame build — Firewall

This build is based off of Northwar’s Sorcerer endgame guide at MaxRoll.gg, and is a general endgame guide capable of finding success in all activities.
</item 6>

<item 7>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Sorcerers are the magic-slinging wizards of Sanctuary in Diablo 4. But with each element they can harness bringing its own unique effect, it can be easy to get
overwhelmed when trying to design a Sorcerer build. That’s where we come in, with our suggestions of the best Sorcerer builds for Diablo 4 season 2.

Sorcerers are capable of harnessing lightning to deal with crowds, ice to keep the hordes at bay, and fire to burn the demons of hell. But they’re also able to
enchant their own spells with unique effects, intricately combining abilities for devastating results.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2. If you need even more information on item affixes or other
more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.

Graph facts:
- 10 scorching-hot discoveries made about the sun in 2023 -MENTIONS-> lightning
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -MENTIONS-> lightning
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> lightning
- Diablo 4 season 2’s biggest draw is its quality-of-life updates -MENTIONS-> Sanctuary
- The best Barbarian builds and skills in Diablo 4 -MENTIONS-> item affixes

Entities: Sanctuary, Sorcerer builds, lightning, ice, fire, item affixes, The best Sorcerer builds and skills in Diablo 4, Sorcerer, Diablo 4, Polygon, season 2
</item 7>

<item 8>
Graph query row:
sorcerer_article: The best Sorcerer builds and skills in Diablo 4
sorcerer_published_on: 2023-10-18T21:43:17+00:00
barbarian_article: The best Barbarian builds and skills in Diablo 4
barbarian_published_on: 2023-10-19T17:16:31+00:00
connection: ['PUBLISHED_BY', 'PUBLISHED_BY']
sorcerer_builds_passages: ['The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nSorcerers are the magic-slinging wizards of
Sanctuary in Diablo 4. But with each element they can harness bringing its own unique effect, it can be easy to get overwhelmed when trying to design a Sorcerer
build. That’s where we come in, with our suggestions of the best Sorcerer builds for Diablo 4 season 2.\n\nSorcerers are capable of harnessing lightning to deal
with crowds, ice to keep the hordes at bay, and fire to burn the demons of hell. But they’re also able to enchant their own spells with unique effects,
intricately combining abilities for devastating results.\n\nIn this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season
2. If you need even more information on item affixes or other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as
well as where to find their original creators.', 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nBest Sorcerer leveling
build\n\nThis build is based off Lexyu’s Sorcerer leveling guide at IcyVeins, and is purely for leveling: i.e., allocating your 58 skill points between levels 1
and 50. If you want more detailed information about the item affixes you want to look for on your gear, or which other Legendary powers are great for your
build, check out the original IcyVeins guide.\n\nBest skills for leveling\n\nThis Sorcerer build guide is going to see you harness the awesome power of
Lightning to speed through the leveling process. There are a few options for Core skills here, but we’re specifically going to be showing you Lexyu’s Chain
Lightning variation. In this build, you’ll also use Frost Bolt, Teleport, Frost Nova, Lightning Spear, and Ice Armor.\n\nHere are the best skills for a Sorcerer
leveling build, organized by the exact order you should purchase them in — bearing in mind that some skills require you to allocate a number of skill points in
the tree in order to unlock them. As such,', 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nAs such, you’ll need to
unlock new skills before enhancing existing ones:\n\nFrost Bolt / Enhanced Frost Bolt Chain Lightning Glinting Frost Bolt Enhanced Chain Lightning / Greater
Chain Lightning Teleport Frost Nova / Enhanced Frost Nova / Mystical Frost Nova Chain Lightning (rank 2-5) Fireball (rank 1 only for the Enchantment slot)
Lightning Spear Glass Cannon (rank 1-3) Devastation Elemental Dominance (rank 1-3) Coursing Currents (rank 1-3) Elemental Attunement Ice Armor / Enhanced Ice
Armor / Shimmering Ice Armor Enhanced Teleport / Shimmering Teleport Enhanced Lightning Spear Vyr’s Mastery Invoked Lightning Spear / Lightning Spear (rank 2-5)
Frost Bolt (rank 2-5) Conduction (rank 1-3) Electrocution (rank 1-3) Align the Elements Protection (rank 1-3) Icy Veil (rank 1-3) Frost Nova (rank 2-3)\n\nBest
Enchantment setup for leveling\n\nThe Enchantment', 'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nFrost Nova (rank
2-3)\n\nBest Enchantment setup for leveling\n\nThe Enchantment slot is key to a ton of your damage as a Sorcerer. You’ll gain two slots as you level, here’s
which spell you want to put in each slot:\n\nLevel 15: Fireball\n\nLevel 30: Chain Lightning\n\nBest Legendary Aspects for Sorcerer leveling build\n\nWhen you
first start your new character, complete these three dungeons first in order to collect their Aspects. Then place them on a piece of gear (ideally jewelry, when
possible). Just keep in mind that you should replace these Legendaries with new versions using the imprint system every few levels.', 'The best Sorcerer builds
and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nThis guide assumes that you’re over level 50, have access to all 58 skill points, and are making
progress on your Paragon Boards. If you want more detailed information about the item affixes you want to look for on your gear, or why these Legendary powers
are great for your build, check out the original MaxRoll guide.\n\nBest skill points for Sorcerer Firewall endgame build\n\nThis build utilizes Firewall as its
core skill and main means of dealing damage. It also uses Fire Bolt, Inferno, Teleport, Ice Armor, and Flame Shield.\n\nSince you need all 58 points for this
build, the order you select these skills in doesn’t matter as long as you have the pre-requisite points in the earlier Skill Tree tiers.', 'The best Sorcerer
builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nHere is how you should spend your Skill Points for this build:\n\nFire Bolt / Enhanced Fire
Bolt / Glinting Fire Bolt\n\nFireball / Enhanced Fireball / Greater Fireball\n\nPotent Warding (rank 1-3)\n\nDevastation\n\nElemental Dominance (rank
1-3)\n\nFlame Shield / Enhanced Flame Shield / Shimmering Flame Shield\n\nTeleport (rank 1-4) / Enhanced Teleport / Shimmering Teleport\n\nIce Armor/Enhanced
Ice Armor\n\nGlass Cannon (rank 1-3)\n\nPrecision Magic (rank 1-3)\n\nAlign the Elements\n\nMana Shield (rank 1-3)\n\nProtection (rank 1-3)\n\nFirewall (rank
1-5) / Enhanced Firewall / Wizard’s Firewall\n\nInner Flames (rank 1-3)\n\nCrippling Flames (rank 1-3)\n\nInferno/Prime Inferno/Supreme Inferno\n\nFiery
Surge\n\nEndless Pyre\n\nCombustion\n\nEnhancement slots for Sorcerer Firewall build\n\nFor this build,', 'The best Sorcerer builds and skills in Diablo 4
(Polygon, 2023-10-18T21:43:17+00:00)\nyou’ll want to use these spells for your two Enchantment Slots:\n\nFirewall Enchantment\n\nFireball\n\nVampiric Powers for
Sorcerer Firewall build\n\nSeason 2 adds the new Vampiric Powers, which allow you to slot five additional abilities into your character to give you new effects.
There are 22 total Vampiric Powers, and each are class agnostic, meaning they’re not unique to any one class.\n\nNorthwar recommends these five Vampiric
Powers:\n\nAnticipation\n\nPrey on the Weak\n\nRavenous\n\nFlowing Veins\n\nMetamorphosis\n\nTo slot those powers you’ll need to have these Pacts available on
your armor:\n\n9 Ferocity\n\n5 Divinity\n\n4 Eternity\n\nLegendary Powers and Uniques for Sorcerer Firewall build\n\nTo make this build work, you’re going to
need powerful Legendary Powers and Unique items to augment your build.', 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)\nNorthwar recommends these items:\n\nHelm : Shared Misery (dungeon drop) or Godslayer Crown (Unique)\n\n: Shared Misery (dungeon
drop) or Godslayer Crown (Unique) Chest : Snowveiled (dungeon drop) or Raiment of the Infinite (Unique)\n\n: Snowveiled (dungeon drop) or Raiment of the
Infinite (Unique) Gloves : Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop)\n\n: Edgemaster’s (dungeon drop), Conceited, or Storm Swell
(dungeon drop) Pants : Disobedience (dungeon drop)\n\n: Disobedience (dungeon drop) Boots : Ghostwalker (dungeon drop), Bounding Conduit (dungeon drop),
Flickerstep (Unique)\n\n: Ghostwalker (dungeon drop), Bounding Conduit (dungeon drop), Flickerstep (Unique) Wand : Edgemaster’s (dungeon drop), Conceited,',
'The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nConceited, or Storm Swell (dungeon drop)\n\n: Edgemaster’s (dungeon
drop), Conceited, or Storm Swell (dungeon drop) Focus : Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop)\n\n: Edgemaster’s (dungeon drop),
Conceited, or Storm Swell (dungeon drop) Amulet : Control (dungeon drop)\n\n: Control (dungeon drop) Ring 1 : Prodigy’s (dungeon drop) or Retribution (dungeon
drop)\n\n: Prodigy’s (dungeon drop) or Retribution (dungeon drop) Ring 2: Engulfing Flames or X’Fal’s Corroded Signet (Unique)\n\nIf you want to get into the
nitty gritty of stat priorities, refer to Northwar’s original guide.', 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)\nrefer to Northwar’s original guide.\n\nParagon Boards for Sorcerer Firewall build\n\nThe Paragon Board system is very complex, and
you’ll want to carefully pair your Glyphs with specific boards for your class:\n\nStarter Board / Reinforced Glyph Burning Instinct / Destruction Glyph Searing
Heat / Adept Glyph Static Surge / Control Glyph Enchantment Master / Exploit\n\nYou should level your Glyphs to 15 in this specific order:\n\nDestruction
Control Adept Exploit Reinforced\n\nYou can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow
Northwar’s path exactly.\n\nHow to make a great Sorcerer build\n\nSorcerers are Diablo 4’s pure caster class, and use elemental magic to differentiate from the
spells of the Druid and Necromancer. The Sorcerer has six Key Passives, but only three elements, which is what you’ll base your build
around:\n\nLightning\n\nFire\n\nIce\n\nLightning builds utilize Lightning skills to deal with large groups of enemies at once.', 'The best Sorcerer builds and
skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)\nWhen making a Lightning build, you’ll end up using the Overflowing Energy Key Passive or the Vyr’s
Master Key Passive, depending on if you’re in melee range or not.\n\nIce builds specialize in slowing enemies down, making it easier to survive their attacks.
Ice builds will take advantage of either the Avalanche Key Passive or the Shatter Key Passive, depending on how much your build relies on crowd control\n\nFire
builds are all about big explosions and burning your enemies. For these builds, you’ll typically use either the Combustion Key Passive for burning enemies or
the Esu’s Ferocity Key Passive for chunky critical strikes.\n\nWhen making your own build for a Sorcerer, you’ll likely end up sticking to one element for your
main damage sources. But that’s not the most important thing you need to remember. Sorcerers die very, very easily in Diablo 4, and they have the weakest
defensive stats by far — however, they can deal a massive amount of damage when built correctly.', 'The best Sorcerer builds and skills in Diablo 4 (Polygon,
2023-10-18T21:43:17+00:00)\nIf you don’t want to end up dead more often than not, make sure to invest heavily in defensive skills while making your way toward
one of the six Key Passives we listed above.\n\nLooking for other Diablo 4 builds? We have explainers on the Barbarian, Rogue, Druid, and Necromancer classes.']
barbarian_builds_passages: ['The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nBarbarians are a relatively simple class in
Diablo 4: They smash stuff. However, Diablo 4 is underscored by math that’s so complex that even the simple act of smashing stuff becomes a dizzying equation.
That’s where we come in, with our suggestions of the best Barbarian builds for Diablo 4 season 2.\n\nBarbarians are big, muscular, and powerful. They excel at
using Bleed to kill their enemies over time. And their unique weapon-swapping system means they get to have more Legendary powers at their disposal than any
other class.\n\nIn this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2. If you need even more information on item
affixes or other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original
creators.', 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nBest Barbarian leveling build\n\nThis build is based off
Lexyu’s Barbarian leveling guide at IcyVeins, and is purely for leveling: i.e., allocating your 58 skill points between levels 1 and 50. If you want more
detailed information about the item affixes you want to look for on your gear, or which other Legendary powers are great for your build, check out the original
IcyVeins guide.\n\nBest skills for leveling\n\nMost Barbarian leveling guides you’ll find make use of Hammer of the Ancients as your primary skill, as it deals
massive damage while remaining very cheap to cast. In this build, you’ll also use Frenzy, Death Blow, Rallying Cry, Leap, and Wrath of the Berserker.\n\nHere
are the best skills for a Barbarian leveling build, organized by the exact order you should purchase them in — bearing in mind that some skills require you to
allocate a number of skill points in the tree in order to unlock them. As such, you’ll need to unlock new skills before enhancing existing', 'The best Barbarian
builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nAs such, you’ll need to unlock new skills before enhancing existing ones:\n\nFrenzy /
Enhanced Frenzy Hammer of the Ancients / Enhanced Hammer of the Ancients / Furious Hammer of the Ancients Combat Frenzy Rallying Cry / Enhanced Rallying Cry /
Tactical Rallying Cry Hammer of the Ancients (ranks 2-3) Leap / Enhanced Leap / Power Leap Hammer of the Ancients (ranks 4-5) Death Blow / Enhanced Death Blow /
Warrior’s Death Blow / Death Blow (rank 2-5) Wrath of the Berserker / Prime Wrath of the Berserker / Supreme Wrath of the Berserker Pit Fighter (ranks 1-3)
Tempered Fury Furious Impulse (ranks 1-3) Walking Arsenal Hamstring No Mercy Expose Vulnerability (ranks 1-3) No Mercy (ranks 2-3) Swiftness (ranks 1-3) Thick
Skin (ranks 1-3) Counteroffensive (ranks 1-3) Defensive', 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\n(ranks 1-3)
Counteroffensive (ranks 1-3) Defensive Stance (ranks 1-3) Slaying Strike (ranks 1-3) Aggressive Resistance Prolific Fury\n\nBest Weapon Expertise setup for
leveling\n\nOnce you unlock the Weapon Expertise system for your Barbarian, you’ll want to place the two-handed sword in your Technique slot. This will cause
all of your Hammer of the Ancients damage to make enemies Bleed.\n\nBest Legendary Aspects for Barbarian leveling build\n\nWhen you first start your new
character, complete these three dungeons first in order to collect their Aspects. Then place them on a piece of gear (ideally jewelry, when possible). Just keep
in mind that you should replace these Legendaries with new versions using the imprint system every few levels.\n\nAspect of Ancestral Force (Sunken Ruins
dungeon in Scosglen) — Hammer of the Ancients quakes outwards,', 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\ndealing
X% of its damage to enemies\n\n(Sunken Ruins dungeon in Scosglen) — Hammer of the Ancients quakes outwards, dealing X% of its damage to enemies Aspect of the
Expectant (Underroot dungeon in Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X%\n\n(Underroot
dungeon in Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X% Edgemaster’s Aspect (Oldstones
dungeon in Scosglen) — Skills deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary
Resource\n\nBarbarian Season 2 endgame build — Double Swing\n\nThis build is based off of AaronActionRPG’s Double Swing Barbarian endgame guide at IcyVeins, and
is a general endgame guide capable of finding success in all activities.', 'The best Barbarian builds and skills in Diablo 4 (Polygon,
2023-10-19T17:16:31+00:00)\nand is a general endgame guide capable of finding success in all activities. This guide assumes that you’re over level 50, have
access to all 58 skill points, and are making progress on your Paragon Boards. If you want more detailed information about the item affixes you want to look for
on your gear, or why these Legendary powers are great for your build, check out the original IcyVeins guide.\n\nBest skill points for Double Swing Barbarian
endgame build\n\nThis build utilizes Double Swing as its core skill. It also uses Steel Grasp, Ground Stomp, Iron Maelstrom, Charge, and Lunging
Strike.\n\nSince you need all 58 points for this build, the order you select these skills in doesn’t matter as long as you have the pre-requisite points in the
earlier Skill Tree tiers. Here is how you should spend your Skill Points for this build:\n\nLunging Strike / Enhanced Lunging Strike / Combat Lunging
Strike\n\nDouble Swing (rank 1-5) / Enhanced Double Swing / Furious Double Swing\n\nEndless', 'The best Barbarian builds and skills in Diablo 4 (Polygon,
2023-10-19T17:16:31+00:00)\nthis build, you’ll want to take One-Handed Mace in your Technique.\n\nVampiric Powers for Double Swing Barbarian build\n\nSeason 2
adds the new Vampiric Powers, which allow you to slot five additional abilities into your character to give you new effects. There are 22 total Vampiric Powers,
and each are class agnostic, meaning they’re not unique to any one class.\n\nAaronActionRPG recommends these five Vampiric Powers:\n\nAnticipation\n\nPrey on
the Weak\n\nRavenous\n\nDomination\n\nMetamorphosis\n\nTo insert those powers you’ll need to have these Pacts available on your armor:\n\n8 Ferocity\n\n3
Divinity\n\n2 Eternity\n\nLegendary Powers and Uniques for Double Swing Barbarian build\n\nTo make this build work, you’re going to need powerful Legendary
Powers and Unique items to augment your build.\n\nAaronActionRPG recommends these items:\n\nHelm : Weapon Master’s Aspect\n\n: Weapon Master’s Aspect Chest :
Aspect of Numbing Wrath (dungeon drop)\n\n: Aspect of', 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nWeapon Master’s
Aspect Chest : Aspect of Numbing Wrath (dungeon drop)\n\n: Aspect of Numbing Wrath (dungeon drop) Gloves : Aspect of Anemia (dungeon drop)\n\n: Aspect of Anemia
(dungeon drop) Pants : Aspect of Might (dungeon drop)\n\n: Aspect of Might (dungeon drop) Boots : Exploiter’s Aspect\n\n: Exploiter’s Aspect Two-handed
Bludgeoning : Aspect of Retribution (dungeon drop)\n\n: Aspect of Retribution (dungeon drop) Two-handed Slashing : Accelerating Aspect\n\n: Accelerating Aspect
Dual Wield 1 : Edgemaster’s (dungeon drop)\n\n: Edgemaster’s (dungeon drop) Dual Wield 2 : Aspect of Berserk Ripping (dungeon drop)\n\n: Aspect of Berserk
Ripping (dungeon drop) Amulet : Earthquake Aspect\n\n: Earthquake Aspect Ring 1 : Aspect of Berserk Fury\n\n: Aspect of Berserk Fury Ring 2: Aspect of Limitless
Rage\n\nIf you want to get into the nitty gritty', 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nRing 2: Aspect of
Limitless Rage\n\nIf you want to get into the nitty gritty of stat priorities, refer to AaronActionRPG’s original guide.\n\nParagon Boards for Double Swing
Barbarian build\n\nThe Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:\n\nStarter Board
/ Bloodfeeder Glyph Decimator / Ambidextrous Glyph Carnage / Might Glyph Flawless Technique / Ire Glyph Weapon Master / Territorial Glyph Bone Breaker / Exploit
Glyph Warbringer / Wrath Glyph\n\nYou should level your Glyphs to 15 in this specific order:\n\nExploit Territorial Ambidextrous Might Wrath Bloodfeeder
Ire\n\nYou can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow AaronActionRPG’s path
exactly.\n\nHow to make a great Barbarian build\n\nBarbarians are Diablo 4’s only pure melee class, with the other classes having hybrid builds for multiple
ranges.', 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nThere are four main ideas that all Barbarian builds stem
from:\n\nFury dumping (most popular in Diablo 4’s pre-season)\n\nWeapon swapping\n\nBleed\n\nBerserk\n\nFury dump builds make use of the Unbridled Rage Key
Passive, which causes your Fury spenders to cost double the Fury but deal more than double the damage. These kinds of builds obviously require very steady Fury
generation, as it’s easy to starve yourself. Berserk builds that use the Unconstrained Key Passive are all about keeping your Barbarian in the Berserk stage at
all times, which nets you a hefty bonus. All Barbarian builds will likely use some skills that make them go Berserk, the Unconstrained Key Passive just enhances
the effect.\n\nBleed builds use the Gushing Wounds Key Passive to create a critical strike heavy build that’s great for whittling down groups of enemies with
damage over time and explosions. Weapon swapping builds offer what is almost certainly the most unique playstyle for Barbarians and takes advantage of the
Walking Arsenal Key Passive.', 'The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)\nThe idea here is that you’re constantly
swapping between your skills and weapons to give yourself massive damage buffs.\n\nWhen making a build, you want to have one of these Key Passives in mind and
choose your skills accordingly. Thankfully skills that share certain aspects (like Bleed) are color-coded on the skill tree, and you can even use the in-game
search function to find them all. Other Barbarian builds let you take advantage of mechanics like Thorns or Shouts. But they still typically end with one of
these Key Passives.\n\nLooking for other Diablo 4 builds? We have explainers on the Sorcerer, Rogue, Druid, and Necromancer classes.']
</item 8>

<item 9>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
refer to Northwar’s original guide.

Paragon Boards for Sorcerer Firewall build

The Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:

Starter Board / Reinforced Glyph Burning Instinct / Destruction Glyph Searing Heat / Adept Glyph Static Surge / Control Glyph Enchantment Master / Exploit

You should level your Glyphs to 15 in this specific order:

Destruction Control Adept Exploit Reinforced

You can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow Northwar’s path exactly.

How to make a great Sorcerer build

Sorcerers are Diablo 4’s pure caster class, and use elemental magic to differentiate from the spells of the Druid and Necromancer. The Sorcerer has six Key
Passives, but only three elements, which is what you’ll base your build around:

Lightning

Fire

Ice

Lightning builds utilize Lightning skills to deal with large groups of enemies at once.

Graph facts:
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Enchantment
- 10 scorching-hot discoveries made about the sun in 2023 -MENTIONS-> lightning
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -MENTIONS-> lightning
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> lightning
- The best Sorcerer builds and skills in Diablo 4 -MENTIONS-> Sanctuary

Entities: lightning, ice, fire, The best Sorcerer builds and skills in Diablo 4, Sorcerer Firewall build, Northwar, Paragon Board system, Glyphs, Starter Board,
Reinforced Glyph, Burning Instinct, Destruction Glyph, Searing Heat, Adept Glyph, Static Surge
</item 9>

<item 10>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
dealing X% of its damage to enemies

(Sunken Ruins dungeon in Scosglen) — Hammer of the Ancients quakes outwards, dealing X% of its damage to enemies Aspect of the Expectant (Underroot dungeon in
Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X%

(Underroot dungeon in Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X% Edgemaster’s Aspect
(Oldstones dungeon in Scosglen) — Skills deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary
Resource

Barbarian Season 2 endgame build — Double Swing

This build is based off of AaronActionRPG’s Double Swing Barbarian endgame guide at IcyVeins, and is a general endgame guide capable of finding success in all
activities.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- Berserker -PART_OF-> Barbarian
- Wild Magic -PART_OF-> Barbarian

Entities: Barbarian, IcyVeins, Hammer of the Ancients, Sunken Ruins dungeon, Scosglen, Aspect of the Expectant, Underroot dungeon, Edgemaster’s Aspect,
Oldstones dungeon, Barbarian Season 2 endgame build — Double Swing, AaronActionRPG, Double Swing Barbarian endgame guide, The best Barbarian builds and skills
in Diablo 4, Diablo 4, Polygon
</item 10>
=== END ITEM id=d267083f2264bebd ===
