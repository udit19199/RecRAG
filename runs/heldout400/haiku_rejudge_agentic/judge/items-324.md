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

This file has 4 items.

=== ITEM id=1c318a7c8f1322dc ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: A man revealed that he made a friendship bracelet.
  Fact 0.1: Friendship bracelets are a fan tradition at Taylor Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: The friendship bracelet seemingly expressed his interest in Swift.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: The man later invited Swift to one of his games on his podcast.
  Fact 1.1: Swift is the Anti-Hero singer.
  Fact 1.2: The game was at Arrowhead Stadium.
  Fact 1.3: Swift accepted the invitation on 24 September.
  Fact 1.4: Swift famously accepted the invitation.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has shared that she has nothing to hide.
  Fact 2.1: Swift is in a new relationship with Kelce.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
Graph query row:
article: Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE
publisher: CBSSports.com
published_on: 2023-10-12T23:21:52+00:00
passage: Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Fantasy Football is all about the matchups. Even though you drafted your team with certain hopes and intentions, your weekly lineup decisions shouldn't be
determined by the order you picked your players in. You need to check who your players play and make sure you've got the right guys in -- and the wrong guys
out.

It's too early to be absolutely sure on which matchups will be easy and which ones will be tough, but we can take some educated guesses based on healthy
personnel, defensive schemes, track records and key details of offenses. The things we know can help us minimize the impact of the things we don't know. This
should lead to better decisions being made.

We'll go through every game and highlight the players who aren't obvious starts and sits (because you don't need to be told to start Christian McCaffrey).
Taylor_Swift: Taylor Swift
relationship: SAID
Travis_Kelce: Travis Kelce
</item 3>

<item 4>
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
</item 4>

<item 5>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> All Too Well
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> All Too Well
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Argentina
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Argentina
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Argentina

Entities: All Too Well, Argentina, Buenos Aires, Karma, Time, Taylor Swift, Travis Kelce, Chiefs, Taylor Swift reveals secret start to her relationship with
Travis Kelce, Eras Tour show, Style, The Independent
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 6>

<item 7>
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
</item 7>

<item 8>
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
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 10>
=== END ITEM id=1c318a7c8f1322dc ===

=== ITEM id=aef26aa0da4e7ffa ===
Gold sentences and facts:
Sentence 0: Rangers also make a lot of choices during character creation (and leveling up) that determine how they play.
  Fact 0.0: Rangers make a lot of choices during character creation.
  Fact 0.1: Rangers make a lot of choices during leveling up.
  Fact 0.2: The choices Rangers make during character creation and leveling up determine how they play.
Sentence 1: During character creation, you must pick the otherworldy patron that you’ve decided to serve.
  Fact 1.0: During character creation, you must pick an otherworldly patron.
  Fact 1.1: The patron you must pick during character creation is the one you have decided to serve.

Retrieved items:
<item 1>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Rangers get Light and Medium Armors and some of the more basic weapons.

Rangers also make a lot of choices during character creation (and leveling up) that determine how they play. At level 1, you’ll choose a Favored Enemy — a
preferred quarry — and a preferred environment that grants you some magical spells.

At level 3, you get to choose a Ranger subclass from Beast Master, Gloom Stalker, or Hunter. Beast Masters get a Companion animal that fights along side you,
Gloom Stalkers focus on stealth, and Hunters get to specialize their melee abilities.

Hunter is the best subclass for a Ranger. Consult our Ranger build guide for more details on why, and to see which skills and feats to pick.

Best class for damage: Barbarian

Barbarians are all about combat. They get light and medium armor (and shields), but they’ve also got the option to keep their AC up even without wearing armor.

Graph facts:
- The best Barbarian builds and skills in Diablo 4 -MENTIONS-> Barbarian
- Wildheart -PART_OF-> Barbarian
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Ranger
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Barbarian
- Berserker -PART_OF-> Barbarian

Entities: Ranger, Beast Master, Gloom Stalker, Hunter, Ranger build guide, Barbarian, How to choose the best class for you in Baldur’s Gate 3, Polygon, Baldur’s
Gate 3
</item 1>

<item 2>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
For Warlocks, it’s best to focus on weapon and armor proficiencies:

Drow ’s Drow Weapon Training adds weapons like the rapiers and hand crossbows

’s Drow Weapon Training adds weapons like the rapiers and hand crossbows Humans ’ and Half-Elves ’ Civil Militia add spears, pikes, halberds, and glaives along
with shields

’ and ’ Civil Militia add spears, pikes, halberds, and glaives along with shields Githyanki Martial Prodigy means you get medium armor along with longswords and
greatswords

Martial Prodigy means you get medium armor along with longswords and greatswords Dwarven Combat Training gets you battleaxes, handaxes, light hammers, and
warhammers. The Shield Dwarf subrace also gets you Medium Armor

When you’re creating and leveling up your Warlock, Charisma (for spellcasting) is your most important stat.

Graph facts:
- The Archfey -PART_OF-> Warlock
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Charisma
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Medium Armor
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Warlock
- The Fiend -PART_OF-> Warlock

Entities: Medium Armor, Charisma, Warlock, The best Warlock subclass, feats, and build in Baldur’s Gate 3, weapon and armor proficiencies, Drow, Drow Weapon
Training, Humans, Half-Elves, Civil Militia, Githyanki, Martial Prodigy, Dwarves, Dwarven Combat Training, Shield Dwarf
</item 2>

<item 3>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Spell Sniper. You learn a cantrip, and the number you need to roll a Critical Hit while attacking is reduced by 1. This effect can stack.

For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, and Wizard classes. If you’re stumped by the choice, consult our guide on what class to start with.

Graph facts:
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Fighter
- Battle Master -PART_OF-> Fighter
- Eldritch Knight -PART_OF-> Fighter
- Champion -PART_OF-> Fighter
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Ranger

Entities: Fighter, Ranger, Barbarian, Paladin, Monk, Cleric, Bard, Wizard, Warlock, The best Warlock subclass, feats, and build in Baldur’s Gate 3, Spell
Sniper, cantrip, Critical Hit, Sorcerer, Rogue
</item 3>

<item 4>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They also get a way to heal themselves and way to get a second attack once per short rest.

At level 3, you get to choose a subclass from Battle Master, Eldritch Knight, and Champion. Battlemasters get a set of Superiority Dice that let you perform
Maneuvers to control the flow of battle and the battlefield. Eldritch Knights add magic into the Fighter’s arsenal. Champions keep it simple with a better
change to deal a Critical Hit (for double damage).

Champion is the best subclass for a Fighter. Consult our Fighter build guide for more details on why, and to see which skills and feats to pick.

Best ranged class for beginners: Ranger

Rangers are agile hunters that focus on Dexterity as their primary Ability. Rangers are an interesting mix of melee, stealth, skills, and magic. That also makes
it harder to get the most out of the class. Rangers get Light and Medium Armors and some of the more basic weapons.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Medium Armor
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Ranger
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Fighter
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Light Armor
- Beast Master -PART_OF-> Ranger

Entities: Fighter, Battle Master, Eldritch Knight, Champion, Superiority Dice, Maneuvers, Fighter build guide, Light Armor, Medium Armor, Weapons, Ranger, How
to choose the best class for you in Baldur’s Gate 3, Critical Hit, Polygon, Baldur’s Gate 3
</item 4>

<item 5>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks.

. Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks. Elemental Adept . Your spells
ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1.

. Your spells ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1. Moderately Armored . You gain Armor
Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20.

. You gain Armor Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20. Spell Sniper.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Fighter
- The Archfey -PART_OF-> Warlock
- The Fiend -PART_OF-> Warlock
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Warlock
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Light Armor

Entities: Warlock, The best Warlock subclass, feats, and build in Baldur’s Gate 3, Elemental Adept, Moderately Armored, Spell Sniper, Polygon, Baldur’s Gate 3
</item 5>

<item 6>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, Warlock, and Wizard classes.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Fighter
- Battle Master -PART_OF-> Fighter
- Eldritch Knight -PART_OF-> Fighter
- Champion -PART_OF-> Fighter
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Ranger

Entities: Fighter, Ranger, Barbarian, Paladin, Monk, Cleric, How to choose the best class for you in Baldur’s Gate 3, Bard, Wizard, Warlock, Sorcerer, Rogue,
Druid, Polygon, Baldur’s Gate 3
</item 6>

<item 7>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Warlock class features

Hit Points

Hit Dice — 1d8 per warlock level

1d8 per warlock level Hit Points at 1st Level — 8 + your Constitution modifier

— 8 + your Constitution modifier Hit Points at higher levels — 5 + your Constitution modifier per warlock level after 1st

Cantrips

The Warlock starts with two cantrips from the Warlock cantrip list.

Baldur’s Gate 3 Warlock Cantrips Cantrip Description Cantrip Description Blade Ward Take only half the damage from Bludgeoning, Piercing, and Slashing attacks
Bone Chill Prevent the target from healing until your next turn. An undead target receives Disadvantage on Attack Rolls. Eldritch Blast Conjure 1 beam(s) of
crackling energy. Friends Gain Advantage on Charisma Checks against a non-hostile creature. Mage Hand Create a spectral hand that can manipulate and interact
with objects. Minor Illusion Create an illusion that compels nearby creatures to investigate.

Graph facts:
- The Fiend -PART_OF-> Warlock
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Fighter
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Warlock
- The Archfey -PART_OF-> Warlock
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Light Armor

Entities: Hit Points, Hit Dice, Cantrips, Constitution, Blade Ward, Bone Chill, Eldritch Blast, Friends, Mage Hand, Minor Illusion, Warlock, The best Warlock
subclass, feats, and build in Baldur’s Gate 3, Polygon, Baldur’s Gate 3
</item 7>

<item 8>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
The Warlock is one of twelve classes in Baldur’s Gate 3. Warlocks are extreme knowledge-seekers who are willing to form a pact with otherwordly beings in order
to gain ancient secrets that’ll bolster their magical arsenal.

To deal with such powerful creatures, Warlocks need to have strong mental power. They need to be able to make a deal with the devil and survive. As such, their
primary ability score is Charisma.

There’s no single best class in Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the
best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to experiment with
multiple classes at once.

Read on to find out everything you need to know about the Warlock class in Baldur’s Gate 3, and learn all about the best Warlock subclass, feats, and build.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Light Armor
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Medium Armor
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Charisma
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Fighter
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> Ranger

Entities: Charisma, Warlock class, The best Warlock subclass, feats, and build in Baldur’s Gate 3, Polygon, Baldur’s Gate 3
</item 8>

<item 9>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you’re bound to The Archfey, you’ll gain an action named “Fey Presence,” which will charm or frighten nearby foes.

The Archfey also has two exclusive spells — Faerie Fire and Sleep.

Best Warlock subclass and build in BG3

Baldur’s Gate 3 races don’t really have any direct impact on your class, but certain races and subraces come with features that play well with certain classes
like extra movement or proficiency with certain weapons.

Graph facts:
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> The Archfey
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> Warlock
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> feats
- The Fiend -PART_OF-> Warlock
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> class

Entities: feats, Warlock, build, The Archfey, Fey Presence, Faerie Fire, races, subraces, class, The best Warlock subclass, feats, and build in Baldur’s Gate 3,
sleep, Polygon, Baldur’s Gate 3
</item 9>

<item 10>
Graph query row:
article: How to choose the best class for you in Baldur’s Gate 3
date: 2023-12-18T17:14:20+00:00
linked_class_or_ranger: subclass
connected_article: Fighter build guide
</item 10>
=== END ITEM id=aef26aa0da4e7ffa ===

=== ITEM id=7cda00de0b9401c2 ===
Gold sentences and facts:
Sentence 0: For now, Griffais explained, it's important to Valve that developers don't need to "split their attention" between different performance targets.
  Fact 0.0: Griffais explained that it is important to Valve that developers do not need to 'split their attention' between different performance targets.
  Fact 0.1: Griffais said this applies for now.
Sentence 1: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 1.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 1.1: Valve has Steam Deck OLED units ready to ship on that same day.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
Graph query row:
article: Steam Deck OLED review: It's just better
organization: developers at Valve
published_on: 2023-11-09T18:00:38+00:00
passage: Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 8>

<item 9>
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
</item 9>

<item 10>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.

Graph facts:
- Steam Deck OLED review: It's just better -MENTIONS-> Steam Deck
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Steam Deck
- The best video games of the year so far -MENTIONS-> Steam Deck
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Steam Deck
- The best Black Friday deals you can get for under $25 -MENTIONS-> Steam Deck

Entities: Steam Deck, Steam Deck OLED, Steam Deck OLED arrival, The Steam Deck OLED arrives November 16 with an improved screen and longer battery life,
Engadget
</item 10>
=== END ITEM id=7cda00de0b9401c2 ===

=== ITEM id=ecdde408d74b4e83 ===
Gold sentences and facts:
Sentence 0: “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.
  Fact 0.0: Zapolsky said that even with those tools, some businesses selling on Amazon might still choose to set prices that are not competitive.
Sentence 1: That means eBay will give you an eBay coupon for the difference in price between what you bought the item for and what the competitor is selling it for, plus an additional 10 percent of that price difference.
  Fact 1.0: eBay will give the buyer an eBay coupon for the difference in price between what the buyer paid for an item and what a competitor is selling it for.
  Fact 1.1: The eBay coupon also includes an additional 10 percent of that price difference.

Retrieved items:
<item 1>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
At the time the EU’s then competition chief, Margrethe Vestager, trumpeted the outcome as setting what she dubbed “new rules for how Amazon operates its
business in Europe” — arguing the settlement meant the ecommerce giant could “no longer abuse its dual role” as marketplace operator and own brand retailer; and
further claiming the outcome would ensure “competing independent retailers and carriers as well as consumers will benefit from these changes opening up new
opportunities and choice.”

Given the commitments the EU already extracted from Amazon after those earlier investigations it’s interesting to see the bloc remains concerned over levers the
ecommerce behemoth still has its disposal to potentially influence competitive outcomes on its marketplace vis-à-vis iRobot vs rival robot vacs.

Since then Amazon has also been named a gatekeeper under the EU’s shiny new ex ante competition regulation, the Digital Markets Act — which lists its
marketplace as a core platform service. This designation means Amazon must comply with a series of up-front obligations, which include restrictions on self
preferencing.

Graph facts:
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- iRobot -MENTIONS-> intense competition
- Commission -SAID-> competition concerns
- A comprehensive list of 2023 tech layoffs -MENTIONS-> iRobot
- iRobot -PARTICIPATED_IN-> lay off 7% of its workforce, roughly 85 employees

Entities: competition concerns, iRobot, EU, Amazon’s iRobot purchase sucks up formal competition concerns in EU, Margrethe Vestager, Amazon’s iRobot purchase,
Amazon marketplace, robot vacs, self preferencing, TechCrunch, settlement, Europe, Amazon, Digital Markets Act
</item 1>

<item 2>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Hart said Amazon sales comprise about 90% of his business, meaning any hit those sales take on Amazon has a considerable impact. He recalled “24 anxious hours”
in September 2022 when a third-party seller of his popular construction toy Brain Flakes listed the toy for a lower price on Target than it was offered on
Amazon.

Molson Hart, CEO of Viahart, an educational toy company that sells on Amazon. Courtesy: Molson Hart

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon -ACCUSED_OF-> FTC antitrust case
- Zapolsky -SAID-> FTC antitrust case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale

Entities: FTC antitrust case, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, Viahart, Molson Hart, Amazon, 24 anxious hours, Brain Flakes,
FTC, CNBC, Target
</item 2>

<item 3>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Below is a list of product(s) in your catalog that are not currently eligible to be the Featured Offer because they are not priced competitively compared to
prices for those products from retailers outside Amazon,” according to the email, which was viewed by CNBC. Unbeknownst to him, Chewy was running a discount
promotion, and dropped the price of his product by a few dollars to $39.99 – less than the $43 offer on Amazon. The algorithm had flagged it as a lower offer,
even though the item on Chewy cost $48.54 after shipping and taxes. Ruckel had to make a choice: Lower the price on Amazon or ask Chewy to raise the price of
his product. He opted for the latter.

Fred Ruckel's company Snuggly Cat makes Ripple Rug, an interactive play mat for cats.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> MediaNews Group
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> The Riverside Press-Enterprise

Entities: Fred Ruckel, Snuggly Cat, Ripple Rug, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, Chewy, FTC, CNBC, Amazon
</item 3>

<item 4>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The seller went on to describe the experience as a “race to the bottom.” “It's long overdue," another commenter wrote. "When they close me down, I'm applying
for a job with the FTC.” Still, others commented that the FTC’s complaint is misguided. “Selling on Amazon is a life-changing opportunity and the amount of
sellers that throw stones at the platform is astounding,” one user wrote.

Seller skepticism

Even sellers who may be sympathetic to the idea of regulating Amazon have concerns, specifically that the FTC's highlighted issues aren't necessarily ones that
would make the seller and consumer experience better. Scott Needham, who sells on Amazon and runs a product-finder tool for other Amazon sellers, said he was
“surprised by some of the points that the FTC selected.”

watch now

“I have over the years been very critical of Amazon,” Needham told CNBC. “I’ve lost a lot of sleep because of some of the things that they have done.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Zapolsky -SAID-> FTC antitrust case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda

Entities: FTC antitrust case, Scott Needham, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, FTC, CNBC, Amazon
</item 4>

<item 5>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“But I’m not sure if this would actually change that.” Scott Moller, an Amazon seller and co-founder of an agency that helps merchants run their storefronts,
said the e-commerce giant has removed some of the challenges that used to be part of running an online business. With FBA, he said, he can ship an item into one
of Amazon’s warehouses for $7.49 per package, while shipping it himself through a traditional carrier would cost him about $12. “I don’t have to have my own
warehouse,” said Moller, who sells grilling accessories on Amazon under the brand Grill Sergeant. “I can use their staff, their storage, and I can instantly
also take the data of advertising, so I can target ads.” He also disputed the FTC’s claim that Amazon has become littered with ads in search results, causing
shoppers to wade through potentially less-relevant products of lesser quality. “We can tailor our ads to hit exactly the consumers we want,” Moller said.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Zapolsky -SAID-> FTC antitrust case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda

Entities: FTC antitrust case, Scott Moller, FBA, Grill Sergeant, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, FTC, CNBC, Amazon,
advertising
</item 5>

<item 6>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Snuggly Cat
- Snuggly Cat -MADE-> Ripple Rug
- Zapolsky -SAID-> FTC antitrust case
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images

Entities: Fred Ruckel, Eastvale, Watchara Phomicinda, MediaNews Group, The Riverside Press-Enterprise, Amazon fulfillment center, Ripple Rug, FTC antitrust
case, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, Getty Images, California, FTC, CNBC, Amazon
</item 6>

<item 7>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The end result of these tactics, the FTC argues, is elevated prices across the web. The company steadily hikes the fees it charges sellers and prevents them
from discounting on other sites, so sellers often inflate their prices off of Amazon, creating an “artificial price floor everywhere,” according to the
complaint. The FTC is seeking to hold Amazon liable for allegedly violating anti-monopoly law, though it has not yet outlined the specific remedies it believes
would best resolve its concerns. In antitrust cases, remedies are often determined only after a court finds the defendant liable. In a blog post, Amazon general
counsel David Zapolsky said third-party sellers set their own prices on the marketplace. The company also invests in tools to help sellers offer “competitive
prices,” he said. “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> MediaNews Group

Entities: David Zapolsky, third-party sellers, anti-monopoly law, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, FTC, CNBC, Amazon
</item 7>

<item 8>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“We can tailor our ads to hit exactly the consumers we want,” Moller said. “It’s a perfect marriage of a transaction, and that’s one of the beauties of what
their marketplace offers.” Needham said he feels he would have been more supportive of the case if it were filed a few years ago, pre-pandemic. At that time, he
said, "I would have felt, yes Amazon is a monopoly… But actually after Covid, into 2023, ecommerce has had a lot of big changes.” He added, "The competition is
just not what it was in 2019." Competitors like Shopify and Walmart are increasingly viable alternatives for many categories of sellers, Needham said, not to
mention rapidly growing Chinese e=commerce companies like Temu. As a result, Needham said he’s seen some significant changes from Amazon. Among those is a
greater ability for Amazon sellers to communicate with buyers, offering select customers certain promotions.
</item 8>

<item 9>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.

Graph facts:
- 5 things we learned so far about the Google antitrust case -MENTIONS-> monopoly power
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Federal Trade Commission -PARTICIPATED_IN-> antitrust lawsuit against Amazon
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda

Entities: Federal Trade Commission, Fred Ruckel, sweeping antitrust lawsuit, monopoly power, Amazon sellers sound off on the FTC's 'long-overdue' antitrust
case, CNBC, Amazon
</item 9>

<item 10>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Needham -SAID-> Fulfillment by Amazon
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda

Entities: Needham, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, Viahart, House committee, FTC lawsuit, Amazon's pricing policies, Amazon
seller fees, Testimony before a House committee, Toys, Molson Hart, Shopify, FTC, CNBC, Amazon, Medium
</item 10>
=== END ITEM id=ecdde408d74b4e83 ===
