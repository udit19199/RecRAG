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

=== ITEM id=a7cfdcfa9ed17034 ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: A friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: The friendship bracelet seemingly expresses his interest in Swift.
Sentence 1: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 1.0: Swift has the Eras Tour booked all through next year.
  Fact 1.1: Swift would rather die than cancel a show.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide.
  Fact 2.1: Swift has a new relationship with Kelce.
  Fact 2.2: Swift has now shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 4>

<item 5>
Graph query row:
report: Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE
published_on: 2023-10-12T23:21:52+00:00
publisher: CBSSports.com
relevant_statements: ['Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com,
2023-10-12T23:21:52+00:00)\n"I was disappointed that she doesn\'t talk before or after her shows because she has to save her voice for the 44 songs she sings
... She doesn\'t meet anybody, or at least she didn\'t want to meet me, so I took it personal," Kelce said on an episode of the "New Heights" podcast he does
with his brother, Philadelphia Eagles center Jason Kelce.\n\nHe revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on
it to give to her, seemingly expressing his interest in Swift. When his brother asked if it was his jersey number or his phone number, Kelce replied, "you know
which one."\n\n"If you\'re up on Taylor Swift concerts, there are friendship bracelets and I had received a bunch of them, but I wanted to give Taylor Swift one
with my number on it."\n\nAug. 5: Kelce names his celebrity crush\n\nWell before the two were rumored to be dating, Kelce was asked his crush.', 'Taylor
Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\nBut before he
could even answer, the answer was given for him.\n\nSept. 14: Rumors aren\'t denied\n\n"The rumors are terrible and cruel, but honey, most of them are true." -
New Romantics\n\nOf course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not
confirm them, but he also did not deny them.\n\nSept. 17: The references begin\n\n"I watched it begin again." - Begin Again\n\nThe rumors began to pick up, and
now there were a lot of people, commentators included, making Swift references around Kelce.\n\nWhen the tight end caught a touchdown pass from Patrick Mahomes,
CBS Sports\' Ian Eagle said, "Kelce finds a blank space for the score."\n\nThis was only the start to many, many, Swift lyrics being referenced during NFL
games.\n\nSept. 21: Kelce sends Swift an invite\n\n"So inviting, I almost jump in."', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored
romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\n21: Kelce sends Swift an invite\n\n"So inviting, I almost jump in." - Gold
Rush\n\nWhile on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.\n\n"It is hilarious how much
traction this has actually got," he said of all the talk. "It\'s like that old game in school called telephone where everybody is just whispering in each others
ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."\n\nHe then admitted that he invited
Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the Chiefs.\n\n"I threw it out there,
I threw the ball in her court and I told her you know, I\'ve seen you rock the stage in Arrowhead you might have to come see me rock the stage in Arrowhead and
see which one is a little more lit. We will see what happens in the near future."', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored
romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\nWe will see what happens in the near future."\n\nSept. 22: Another
bracelet\n\n"Lately [he\'s] been dressing for revenge." - Vigilante Sh--\n\nKelce was spotted at his nonprofit charity event wearing a bracelet that says
"Vigilante Sh--," the name of one of Swift\'s songs.\n\nSept. 24: Swift attends Chiefs game\n\n"Dating the boy on the football team." - Fifteen\n\nSwift took
Kelce up on his offer and attended the Chiefs\' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering and even
chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0. Kansas
City went on to win 41-10 to improve to 2-1.\n\nSept. 24: Swift mingles with Donna Kelce\n\n\n\n"Talkin\' over a football team."', 'Taylor Swift-Travis Kelce
timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\n24: Swift mingles with Donna
Kelce\n\n\n\n"Talkin\' over a football team." - Mean\n\nSwift already seems to be making friends with the family. Swift was sitting next to Donna in the box and
the two were seen talking and laughing as the game went along.\n\nSept. 24: Postgame interviews turn into Swiftie talk\n\nThe postgame press conference seemed
more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the relationship and how Swift ended up at
Arrowhead.\n\n"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There\'s some things with Trav where he kind of just says
it and you don\'t know if it\'s true or not. He says it so calmly. ... Friday he was like, \'Yeah, I think she\'s coming to the game this weekend,\' and then
moved about his business."', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com,
2023-10-12T23:21:52+00:00)\nHead coach Andy Reid was also asked about his star tight end\'s supposed relationship, to which he quipped, "I set them
up."\n\nSept. 24: Kelce and Swift leave stadium together\n\n"You know when it\'s time to go." - It\'s Time To Go\n\nThis was the first time the two were
photographed together. They walked side by side, Kelce fittingly wearing an outfit with "1989" in the title and Swift wearing a Chiefs zip-up.\n\nSept. 24: The
two leave in a convertible\n\n"Nothing good starts in the getaway car." - Getaway Car\n\nSwift and Swift were spotted in a convertible, with the tight end
driving, after the Chiefs\' win. Kelce reportedly rented out a restaurant for a private party with Swift and his teammates.\n\n"Taylor Swift likes that Travis
Kelce \'pursued her\' and the two are having a \'great time getting to know each other,\'" a source told Entertainment Tonight.', 'Taylor Swift-Travis Kelce
timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\n"\'Taylor wants a guy that\'s into
his career, does his own thing, but is also supportive and understands her,\' the source added, "and Travis fits those qualities.\'"\n\nSept. 25: Kelce jersey
sales skyrocket\n\nCall it what you want; I\'ll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey
sales, his podcast is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game
last week, according to Front Office Sports.\n\nThere has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.\n\nSept.
25: Belichick weighs in on the matter\n\n\n\nNew England Patriots head coach does not always comment on situations that take place in the game, but he was fully
ready with a sound bite on the Kelce-Swift situation.', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs
TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\n"Well I would say that Travis Kelce\'s had a lot of big catches in his career. This would be the
biggest."\n\nSept. 26: Chiefs games get even more popular\n\n"Welcome to New York (*New Jersey), it\'s been waiting for you." - Welcome to New York\n\nThe
defending Super Bowl champions are already one of the more popular teams in the league, but it is becoming the hottest ticket in town, no matter what town that
is.\n\n"Since Swift\'s appearance at the Chiefs game on Sunday, the Jets have sold more tickets for their game against the Chiefs in a single day than since the
season started -- by more than double," a representatives at StubHub told to Town & Country magazine.\n\nSept. 27: Mahomes confirms he met Swift\n\n"It was
enchanting to meet you, all I can say is I was enchanted to meet you."', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between
pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\n- Enchanted\n\nMahomes said he had not met Swift when asked in the postgame press conference,
but he did meet her at the afterparty.\n\n"Yea I met her. She\'s really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma
let them have their privacy and just keep it moving."\n\nSept. 27: Kelce breaks his silence\n\n"And they said speak now." -Speak Now\n\nKelce addressed Swift\'s
attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was all excited that she was
there -- that s--- was absolutely hysterical. It\'s definitely a game I\'ll remember, that\'s for damn sure."\n\nHe is not going to speak too much more about
the relationship, noting that he wants "to respect both of our lives."\n\n"Yeah, my personal life -- that\'s not so personal.', 'Taylor Swift-Travis Kelce
timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\n"Yeah, my personal life -- that\'s
not so personal. I did this to myself Jason, I know this. What\'s real is that you know it is my personal life. ... She\'s not in the media as much as I am
doing this show every single week."\n\nSept. 27: Jerry Jones comments on the Swift effect\n\nSwift is getting the owners talking, too.\n\n"It is hard to move
that needle dramatically on a great feeling, a great attraction in the NFL," Jones said Wednesday on 105.3FM The Fan in Dallas. "We get a lot of eye balls. She
blew it off the charts. It is amazing. It makes you smile. But it just shows the depth of her following, especially with her core constituency. We all, I think,
felt the same way when she\'s sitting up there with [Travis Kelce\'s] mom and she\'s going nuts when he\'s playing and those hearts are popping out
everywhere.', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com,
2023-10-12T23:21:52+00:00)\nHow do you not get that? That\'s called images right there."\n\nOct. 1: Swift attends Chiefs-Jets\n\n"Welcome to New York, it\'s
been waiting for you," - Welcome to New York\n\nOK, so the Jets technically play in New Jersey, but we\'re just basing this off their name. Swift went to her
second consecutive Chiefs game, showing up with some famous friends, Blake Lively, Ryan Reynolds and Hugh Jackman. In the week leading up to the game, there
were reports that the singer would be at the game. The Chiefs defeated the Jets 23-20.\n\nOct. 5: NFL defends itself\n\nSwift coverage is everywhere and the
league even changed their social media bio to one of the singer\'s lyrics. The league\'s X (formerly Twitter) bio to read, "NFL (Taylor\'s Version)," while
their Instagram account read, "Chiefs are 2-0 as Swifties."', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star,
Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\nSome found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting
as such.\n\n"We frequently change our bios and profile imagery based on what\'s happening in and around our games, as well as culturally," the NFL wrote in a
statement. "The Taylor Swift and Travis Kelce news has been a pop cultural moment we\'ve leaned into in real time, as it\'s an intersection of sport and
entertainment, and we\'ve seen an incredible amount of positivity around the sport."\n\nOct. 6: Travis and Jason think the NFL is doing too much\n\nThe NFL is
really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.\n\n"I think it\'s fun when they show who\'s at the
game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what you\'re watching. But at the same time, they\'re
overdoing it a little bit."', 'Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com,
2023-10-12T23:21:52+00:00)\nBut at the same time, they\'re overdoing it a little bit."\n\nJason also chimed in on the matter, saying, "I just think the NFL is
not used to celebrities coming to the games. Like basketball has it all figured out. They\'re all courtside, they\'re sitting there. They show \'em once or
twice, but then they get back to the game."\n\nOct. 12: Swift attends Chiefs-Broncos\n\n"Now it\'s too late for you and your white horse *Bronco, to come
around" - White Horse\n\nOne day after her movie premiere, Swift is in the house for the Chiefs\' "Thursday Night Football" divisional matchup against the
Broncos. The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will
have minimal Swift coverage, though so far there has already been a mention of the superstar.\n\nOct.', 'Taylor Swift-Travis Kelce timeline: Everything to know
about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)\nOct. 15: Kelce, Swift make \'SNL\' cameos\n\nThe two were spotted
out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift was getting during
games, while the singer introduced the musical act of the night, Ice Spice.\n\nOct. 22: Swift attends fifth Chiefs game\n\nSwift has now attended games
broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and will be shown on CBS. She is
sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes\' wife, Brittany.']
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Jason, Taylor Swift, rumored romance, Travis Kelce, pop cultural moment, Swift-Kelce coverage, Taylor Swift-Travis Kelce timeline: Everything to
know about rumored romance between pop star, Chiefs TE, CBSSports.com
</item 8>

<item 9>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.

Graph facts:
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Time
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: Time, Taylor Swift, Eras Tour, Travis Kelce, Taylor Swift reveals secret start to her relationship with Travis Kelce, The Independent
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a7cfdcfa9ed17034 ===

=== ITEM id=bfa9e58824f2022c ===
Gold sentences and facts:
Sentence 0: These delays may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.
  Fact 0.0: Delays may lead to bets being refunded.
  Fact 0.1: Whether bets are refunded depends on the circumstances and the specific rules of the sportsbook.
Sentence 1: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 1.0: Every sportsbook adjusts when a betting line moves or odds change for a legitimate reason.
  Fact 1.1: Examples of legitimate reasons include an injury or a roster move.
Sentence 2: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 2.0: Sportsbooks are also gathering this information.
  Fact 2.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.

Retrieved items:
<item 1>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Once the season has started and you get to see how the various rookies are performing, you can gather the relevant data. But sportsbooks are also gathering this
information, and they begin to tighten the NBA Rookie of the Year lines in response.

So, while you can better guess which player will hoist the Wilt Chamberlain Trophy, the odds will be tighter, and your potential return on your wager will be
reduced.

Pros and cons of betting early on the NBA Rookie of the Year:

Pros:

You have an opportunity to find better value.

The betting odds are fresh and haven't been adjusted.

Cons:

There is a much higher risk as a player's performance, health, and team dynamic can change throughout the season.

You’ll find limited information to make an informed bet.

Pros and cons of waiting to bet on the NBA Rookie of the Year:

Pros:

There is a lot more information available to make an informed bet.

You don’t risk betting on a player not remotely in contention.

Cons:

There is a reduced opportunity to find value.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA Rookie of the Year award

Entities: Wembanyama, NBA Rookie of the Year, Wilt Chamberlain Trophy, NBA Rookie of the Year Odds - Wembanyama the early favorite, Sporting News, NBA, betting
odds
</item 1>

<item 2>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Look for a quality welcome bonus for new players and attractive promotions for existing customers.

Also known for being in the DFS market, FanDuel has one of the largest followings in the country. It offers competitive odds for NBA ROTY and has one of the
best sports betting apps. Look for a quality welcome bonus for new players and attractive promotions for existing customers. BetMGM Sportsbook: As one of the
most recognizable names in the gambling industry, BetMGM knows how to attract and keep customers with competitive odds for all bet types, including futures bets
and the NBA Rookie of the Year. BetMGM offers many deposit and withdrawal options and 24/7 customer service, and generous sports betting bonuses and promotions.

As one of the most recognizable names in the gambling industry, BetMGM knows how to attract and keep customers with competitive odds for all bet types,
including futures bets and the NBA Rookie of the Year. BetMGM offers many deposit and withdrawal options and 24/7 customer service, and generous sports betting
bonuses and promotions.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA Rookie of the Year award

Entities: Wembanyama, NBA Rookie of the Year, NBA Rookie of the Year Odds - Wembanyama the early favorite, BetMGM, Sporting News, FanDuel, BetMGM Sportsbook,
NBA, sports betting
</item 2>

<item 3>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Season Player Position Team 2023 Paolo Banchero Forward Orlando Magic 2022 Scottie Barnes Forward Toronto Raptors 2021 LaMelo Ball Guard Charlotte Hornets 2020
Ja Morant Guard Memphis Grizzlies 2019 Luka Dončić Guard/Forward Dallas Mavericks 2018 Ben Simmons Forward/Guard Philadelphia 76ers 2017 Malcolm Brogdon Guard
Milwaukee Bucks 2016 Karl-Anthony Towns Center Minnesota Timberwolves 2015 Andrew Wiggins Forward/Guard Minnesota Timberwolves 2014 Michael Carter-Williams
Guard Philadelphia 76ers 2013 Damian Lillard Guard Portland Trail Blazers 2012 Kyrie Irving Guard Cleveland Cavaliers 2011 Blake Griffin Forward Los Angeles
Clippers 2010 Tyreke Evans Guard/Forward Sacramento Kings

What other NBA futures can I bet on?

Rookie of the Year is not the only form of NBA futures betting.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA MVP Odds 2024 Season - Jokic and Giannis early favorites -MENTIONS-> NBA futures betting
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs

Entities: NBA futures betting, Wembanyama, NBA Rookie of the Year Odds - Wembanyama the early favorite, Paolo Banchero, Orlando Magic, Scottie Barnes, Toronto
Raptors, LaMelo Ball, Charlotte Hornets, Ja Morant, Memphis Grizzlies, Luka Dončić, Dallas Mavericks, Ben Simmons, Philadelphia 76ers
</item 3>

<item 4>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
DraftKings Sportsbook: DraftKings, known as a daily fantasy sports operator before becoming a renowned sportsbook, offers up a nice variety of futures bets,
including the NBA Rookie of the Year award. The online sportsbook and betting app are intuitive and offer an easy-to-understand betting experience.
Additionally, DraftKings fields a competitive welcome bonus for new players.

DraftKings, known as a daily fantasy sports operator before becoming a renowned sportsbook, offers up a nice variety of futures bets, including the NBA Rookie
of the Year award. The online sportsbook and betting app are intuitive and offer an easy-to-understand betting experience. Additionally, DraftKings fields a
competitive welcome bonus for new players. FanDuel Sportsbook: Also known for being in the DFS market, FanDuel has one of the largest followings in the country.
It offers competitive odds for NBA ROTY and has one of the best sports betting apps. Look for a quality welcome bonus for new players and attractive promotions
for existing customers.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA futures betting

Entities: NBA Rookie of the Year award, Wembanyama, NBA Rookie of the Year Odds - Wembanyama the early favorite, Sporting News, DraftKings, FanDuel
</item 4>

<item 5>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Caesars Sportsbook: Caesars knows its way around sports betting when you use its app and online sportsbook. It offers favorable odds for almost every bet type,
including NBA ROTY betting, and offers a nice variety of deposit and withdrawal options. Additionally, Caesars has a competitive welcome bonus for new players
and runs NBA betting promos for existing players.

When is the best time for NBA ROTY odds?

The earlier you bet on NBA ROTY, the better the odds. But you’ll be doing so without much information that could be key in making an educated decision.

Conversely, if you wait for the data you need for an informed bet, the odds will tighten up, and you will see less of a return on your wager.

When the lines are first released for NBA ROTY honors, the season hasn’t even started yet, so there are no statistics, trends, or player news. The pre-season
odds are so favorable because even the oddsmakers don’t have the data to craft more specific lines.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA Rookie of the Year award

Entities: Wembanyama, NBA Rookie of the Year betting, NBA Rookie of the Year Odds - Wembanyama the early favorite, Caesars, Sporting News, Caesars Sportsbook,
NBA
</item 5>

<item 6>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Rookie of the Year is not the only form of NBA futures betting. Legal online sportsbooks have a wide selection of NBA futures you can bet on, including:

In addition to futures bets, you can wager on many NBA bets during the regular season and postseason. NBA game lines (moneylines, point spreads, and totals) are
the most popular, but you’ll also be able to bet on NBA team and player props, parlays, and more.

The NBA is the second-most popular odds market to bet on in the US, behind NFL betting odds. That means every online betting site always has a great selection
of NBA odds.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Wembanyama, NBA Rookie of the Year Odds - Wembanyama the early favorite, US, Sporting News, Rookie of the Year, NBA
</item 6>

<item 7>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Cons:

There is a reduced opportunity to find value.

The odds may not be as favorable.

Keep in mind that it’s always important to take into consideration your risk tolerance while you weigh the potential rewards. No matter when you decide to make
your wager, always gather as much information as possible before risking your hard-earned money.

Who are the past NBA ROTY winners?

For context, here is a list of the National Basketball Association Rookie of the Year winners since 2010.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA Rookie of the Year award

Entities: Wembanyama, NBA Rookie of the Year, NBA Rookie of the Year Odds - Wembanyama the early favorite, Sporting News, National Basketball Association
</item 7>

<item 8>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
We urge you to read the requirements for any available promotions and offers before accepting them so you’re aware of any restrictions before betting.

You can now place your initial deposit, which is easy at online betting sites thanks to various deposit and withdrawal options.

To bet on the NBA Rookie of the Year, navigate to the basketball section of the sportsbook website, select the NBA, and go to the futures tab.

You’ll find Rookie of the Year odds, NBA MVP odds, and more there. Select the bet you want to make, enter the amount you want to bet, and finalize your bet
slip.

Ensure you’re using only legal online sportsbooks for any type of betting. Offshore sportsbooks advertise widely on the internet, but they are illegal and offer
no protection against fraud and other criminal activities.

Best sportsbooks for NBA Rookie of the Year odds

Here are some of the best sports betting sites where you can bet on the next NBA Rookie of the Year.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA Rookie of the Year award

Entities: Wembanyama, NBA Rookie of the Year, NBA Rookie of the Year Odds - Wembanyama the early favorite, Sporting News, NBA MVP, NBA
</item 8>

<item 9>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
His wingspan and defensive prowess will challenge opponents. Offensively, he capitalizes on mismatches and has diverse scoring abilities, from the paint to the
3-point line. His underrated passing syncs well with OKC's pace and style.

How to bet on NBA Rookie of the Year

The first thing you need when preparing to bet on the NBA Rookie of the Year Award, or any other bet, is an account at an online sportsbook.

Signing up for a new betting account doesn’t take much time. Once you visit the online sportsbook you’re interested in, click a “sign up” or “register” button.
This will take you to a form for your details. Fill that out and accept the sportsbook’s terms and conditions.

At that point, you’ll be able to claim your welcome bonus, which you’ll qualify for by following the link on this page from our site to the online sportsbook.
If you need to enter a code to get the bonus, you’ll also find that code on our site.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Gregg Popovich
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Victor Wembanyama
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NFL
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> NBA futures betting

Entities: NBA Rookie of the Year award, Wembanyama, OKC, NBA Rookie of the Year Odds - Wembanyama the early favorite, Sporting News, NBA, online sportsbook
</item 9>

<item 10>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.

Safety and Security:

Top NBA betting sites prioritize users' safety.

Ensure your personal information is secure.

Some sportsbooks excel in ensuring cybersecurity; always go for those with solid reputations.

How to find the best NBA lines

Not all sportsbook betting odds are created equal. If you're willing to put in some legwork, you can shop for the best NBA lines to maximize your potential
return.

Movement in the odds depends on multiple factors, including team news, injuries, public betting, recent play, and even the number of games a team plays in a
row.

Oddsmakers use all the available information to craft their lines, and you can find value in shopping around for the most favorable lines and odds.

You can use our NBA odds page (see link above) to find the best lines for your prospective bet, or you can open accounts at multiple sportsbooks and do the
shopping yourself.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA betting, NBA, sportsbooks, responsible gaming, The Best NBA Betting Sites and Apps for the 2023-24 Season, cybersecurity
</item 10>
=== END ITEM id=bfa9e58824f2022c ===
