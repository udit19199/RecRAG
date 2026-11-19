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

=== ITEM id=bbb1b174b066cb1e ===
Gold sentences and facts:
Sentence 0: In July, speaking on the podcast he shares with his brother, New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City.
  Fact 0.0: In July, Kelce spoke on the podcast he shares with his brother, New Heights.
  Fact 0.1: Kelce said he intended to give Swift a friendship bracelet with his number on it.
  Fact 0.2: Kelce said he intended to give the bracelet during Swift's Eras Tour concert in Kansas City.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: On his podcast, Kelce later invited the 'Anti-Hero' singer to one of his games at Arrowhead Stadium.
  Fact 1.1: The 'Anti-Hero' singer famously accepted the invitation on 24 September.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the
classic game “Kiss, Marry, Kill”. Speaking with AfterBuzz TV, Kelce was given the options of Swift, Katy Perry and Ariana Grande. He said he’d ultimately kill
Grande, marry Perry and kiss Swift. That was about six years ago, but it has more recently set tongues wagging as fans wondered how long the plan for them to
meet has been in the works. Cruel Summer Fast-forward seven years, and things started to heat up. In July, speaking on the podcast he shares with his brother,
New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City. Unfortunately, the
bracelet never made it to her wrist or his digits to her phone.

Graph facts:
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Kansas City
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Kansas City
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Kansas City
- Beyoncé tops box office with $21 million debut for concert film ‘Renaissance’ -MENTIONS-> Kansas City
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Kansas City

Entities: Kansas City, Eras Tour concert, When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover, Kelce, Super Bowl victory, Kiss, Marry,
Kill, AfterBuzz TV, friendship bracelet, Taylor Swift, New Heights, The Age, sport, Katy Perry, pop culture, Ariana Grande
</item 5>

<item 6>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Could they have been driving off into the sunset? Of course, none of this actually confirms Swift and Kelce are actually dating. But regardless of whether it’s
a friendship or a romantic relationship, it has sent shockwaves through two massive cultural franchises, expanding the NFL’s audience into uncharted territory.
It’s a cultural pairing on par with – and perhaps even bigger than – Kim Kardashian and Kanye West.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover, Kelce, Kanye West, Taylor Swift, Kim Kardashian, The Age
</item 6>

<item 7>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Anti-Hero
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Time
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Anti-Hero
- Taylor Swift -MENTIONS-> Instagram

Entities: Anti-Hero, Time, Taylor Swift, Travis Kelce, relationship, Taylor Swift reveals secret start to her relationship with Travis Kelce, paparazzi, media,
The Independent - Life and Style
</item 7>

<item 8>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Jason Kelce
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Jason Kelce
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Jason Kelce
- Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jason Kelce
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Jason Kelce

Entities: Jason Kelce, Taylor Swift, Travis Kelce, New York Jets, The Independent, Kansas City Chiefs, National Football League, game against the New York Jets,
New Heights with Jason and Travis Kelce, Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage
</item 8>

<item 9>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Time, Taylor Swift, NFL fans, NFL appearance, Taylor Swift is Time’s Person of the Year, Travis Kelce, Chiefs game, Eras Tour show, Kansas City
Chiefs, The Independent - Life and Style, football
</item 9>

<item 10>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?

In August, the brothers discussed Kelce’s moustache, which he grows every year in honour of the Chief’s head coach, Andy Reid. When asked if he knew what Swift
thought of the ’stache, he remained cheekily confident. “We’re not going to bring up Taylor Swift in this episode,” he says, “But something tells me she’s going
to like it.” There was a clear suggestion Kelce had made contact with Swift, and was even going to meet her. Shortly after that remark, Kelce took a week off
training to visit New York City, according to Vulture. Coincidentally, this happened to be while Swift was re-recording her 1989 tracks in the city. Welcome to
New York

The suspicion the two were actually spending time together in New York was seemingly confirmed in September when it was reported they had been “quietly hanging
out” for months.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> NFL
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Kansas City
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Eras Tour concert
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Jaguars

Entities: When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover, Andy Reid, Vulture, New York City, visit to New York City, re-recording
1989 tracks, quietly hanging out, Taylor Swift, Travis Kelce, The Age, Chiefs
</item 10>
=== END ITEM id=bbb1b174b066cb1e ===

=== ITEM id=100eedfe9fc49a38 ===
Gold sentences and facts:
Sentence 0: This isn’t the first time that Jada has opened up about her long-term commitment to her husband.
  Fact 0.0: This is not the first time Jada has opened up about her long-term commitment to her husband.
Sentence 1: Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus.
  Fact 1.0: Jada praised her children for how they helped her grow.
  Fact 1.1: Jada said: 'My children, they're little gurus.'
Sentence 2: She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep sense of self-acceptance”.
  Fact 2.0: Jada and Will have two children.
  Fact 2.1: Jada and Will's children are Jaden, who is 25, and Willow, who is 22.
  Fact 2.2: Jada praised her children for being her 'little gurus' throughout this time.
  Fact 2.3: Jada's children taught her 'a deep sense of self-acceptance'.

Retrieved items:
<item 1>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
The altercation came after Rock said: “Jada, I love you. GI Jane 2, can’t wait to see you,” referring to Jada’s shaved head. While accepting the award for Best
Actor, Will became teary as said he was a “fierce defender of his family”.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> separation
- Jacob Lucas -SAID-> separation
- Susan Winter -SAID-> separation
- Winter -SAID-> separation
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -PUBLISHED_BY-> The Independent - Life and Style

Entities: separation, A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation, Jada Pinkett Smith, Rock, altercation, GI Jane 2, award
for Best Actor, Will Smith, The Independent
</item 1>

<item 2>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Will Smith has spoken out about his “tumultuous” relationship with his wife, Jada Pinkett Smith, while speaking to a crowd of fans in Baltimore, Maryland.

The 55-year-old actor took the stage at the Enoch Pratt Free Library on 18 October, as he made an appearance at a press event for his wife’s book, Worthy,
alongside their two children – Jaden, 25, and Willow, 22.

Graph facts:
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Willow
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Jada Pinkett Smith
- Willow -PARTICIPATED_IN-> Willow’s birth
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Willow
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Jada Pinkett Smith

Entities: Willow, Jada Pinkett Smith, Will Smith, Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith, Jaden, Enoch Pratt Free
Library, press event for Worthy, Worthy, The Independent, Baltimore, Maryland
</item 2>

<item 3>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
During an interview with Hoda Kotb for Today, which aired on 13 October, the actor confirmed that she and her partner have been living “completely separate
lives” for the past seven years. However, according to Jada, their choice to live in separate homes doesn’t mean that they’ll be getting divorced anytime soon.

“I made a promise that there will never be a reason for us to get a divorce,” she said. “We will work through whatever. And I just haven’t been able to break
that promise.”

When asked why she and Will didn’t “just amicably divorce,” Jada simply replied: “We don’t want to.”

She then explained how her family shaped that decision, as she and the King Richard star share two children - Jaden, 25, and Willow, 22. Will also has a
30-year-old son, Trey, who he shares with ex-wife Sheree Zampino. “We love our family and we love each other.
</item 3>

<item 4>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
However, during an appearance on the Today show on 16 October, Jada clarified to Hoda Kotb that she and the Suicide Squad actor are “working hard” and are
“concentrating on healing” to get their relationship to a better place.

“There’s no finding another great love, and I think that’s the point,” she said. “It’s like we are in a place now that we are in a deep, healing space. And we
are really concentrating on healing the relationship between us.”

The Girls Trip star continued: “There’s no divorce on paper. We really have been working hard. That’s the whole thing. We are working very hard at bringing our
relationship together. Back to a life partnership.”

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Jada Pinkett Smith
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Jada Pinkett Smith
- Jada Pinkett Smith reveals why she and Will Smith never signed a prenup -MENTIONS-> Jada Pinkett Smith
- Jada Pinkett Smith -SAID-> separation
- Jada Pinkett Smith -SAID-> self-acceptance

Entities: Jada Pinkett Smith, Hoda Kotb, Will Smith, Suicide Squad, Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith, Today
show, Girls Trip, appearance on the Today show, relationship, The Independent - Life and Style
</item 4>

<item 5>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Prior to sharing the video, Smith showed his support for his partner and her success in a statement toThe New York Times. During the interview, the outlet noted
that Smith said his wife’s memoir “kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” he added.

In an interview with People published on 10 October, Jada first revealed that she and her husband were living separately. “We’ve been doing some really
heavy-duty work together,” she said. “We just got deep love for each other and we are going to figure out what that looks like for us.”

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Jada Pinkett Smith, Will Smith, Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith, Interview with
People, The Independent, People
</item 5>

<item 6>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Less than two weeks later, Jada and Will appeared on an episode of Jada’s Red Table Talk where Jada confirmed that she did have a relationship with Alsina,
saying that she and Will were separated at the time.

“We decided that we were going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,”
Jada said at the time.

“From there, as time went on, I got into a different kind of entanglement with August... Yes, it was a relationship absolutely.”

Jada also said that Will didn’t give permission, as the only person who could give her permission was herself.

2021: Jada discusses her and Will’s sex life, Will’s memoir is released

In another episode of Red Table Talk in October 2021, Jada said it “takes work” to maintain a spark in the bedroom.

“It’s hard... the thing Will and I talk about a lot is the journey.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> separation
- Jacob Lucas -SAID-> separation
- Susan Winter -SAID-> separation
- Winter -SAID-> separation
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -PUBLISHED_BY-> The Independent - Life and Style

Entities: separation, A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation, August Alsina, Jada Pinkett Smith, relationship with
August Alsina, permission, maintaining a spark in the bedroom, episode of Red Table Talk, Will Smith, Red Table Talk, The Independent
</item 6>

<item 7>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
In an interview with People on 10 October, Jada shared that during the 2022 Academy Awards - when Will slapped Chris Rock after he cracked a joke about Jada’s
appearance - the pair had actually been separated for six years. “We’ve been doing some really heavy-duty work together,” she explained, while promoting her new
memoir, Worthy. “We just got deep love for each other and we are going to figure out what that looks like for us.”

She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep
sense of self-acceptance”. Although the married couple are not living together, she insisted that their marriage has never been an open one, as she and Will
have “eliminated the chance of betrayal” while going on to live their “separate lives”.

Graph facts:
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> 2022 Academy Awards
- Jada Pinkett Smith reveals why she and Will Smith never signed a prenup -MENTIONS-> Chris Rock
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Chris Rock
- Chris Rock -PARTICIPATED_IN-> 94th annual Oscars
- Jada Pinkett Smith reveals why she and Will Smith never signed a prenup -MENTIONS-> 2022 Academy Awards

Entities: Chris Rock, 2022 Academy Awards, separation, self-acceptance, Willow, Jada Pinkett Smith, Jada Pinkett Smith and Will Smith have been separated since
2016 - when should couples just get divorced?, Will Smith, Jaden, Worthy, The Independent - Life and Style, People
</item 7>

<item 8>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“We’ve been doing some really heavy-duty work together,” Jada said. “We just got deep love for each other and we are going to figure out what that looks like
for us.”

The pair have been married since 1997 and share son Jaden, 25, and daughter Willow, 22. The Fresh Prince of Bel-Air star also has a son Trey, 30, whom he shares
with his first wife, Sheree Zampino.

As for her children, Jada praised them for how they’ve helped her grow. “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance,”
she told the outlet.

Jada and Will have often opened up about their relationship over the years, and have had very public ups and downs throughout their marriage. Here’s a timeline
of their relationship.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> separation
- Jacob Lucas -SAID-> separation
- Susan Winter -SAID-> separation
- Winter -SAID-> separation
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> self-acceptance

Entities: separation, Trey, Fresh Prince of Bel-Air, self-acceptance, deep love, Sheree Zampino, A timeline of Will Smith and Jada Pinkett Smith’s relationship
amid separation, Willow, Jada Pinkett Smith, Will Smith, Jaden, The Independent - Life and Style
</item 8>

<item 9>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance.”

On 16 October, Will broke his silence on his wife’s comments during an interview with the New York Times. The outlet noted that he said his partner’s memoir
“kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” Will added.

Graph facts:
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Jada Pinkett Smith
- Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith -MENTIONS-> Jada Pinkett Smith
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Jada Pinkett Smith
- Jada Pinkett Smith -SAID-> separation
- Jada Pinkett Smith -SAID-> self-acceptance

Entities: Jada Pinkett Smith, Will Smith, interview, prenup, Jada Pinkett Smith reveals why she and Will Smith never signed a prenup, New York Times, memoir,
The Independent
</item 9>

<item 10>
Graph query row:
first_report: Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced?
first_published: 2023-10-16T13:41:30+00:00
second_report: Jada Pinkett Smith reveals why she and Will Smith never signed a prenup
second_published: 2023-10-16T15:48:10+00:00
report_connection: None
report: Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced?
person: Willow
relationship: MENTIONS
passage: Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Longtime Hollywood couple Will Smith and Jada Pinkett Smith sent shockwaves around the world this week when the Red Table Talk host revealed that she and the
actor have been separated since 2016. Both Jada and Will have previously doubled down on their stance against getting a divorce, but her recent reveal about
their lengthy separation poses a new question: When should couples just get a divorce?
</item 10>
=== END ITEM id=100eedfe9fc49a38 ===

=== ITEM id=04ae99cd6857d424 ===
Gold sentences and facts:
Sentence 0: 10:03 p.m.: Love hits Dillon out in the front, who makes his defender miss before racing beyond the marker for a first down.
  Fact 0.0: At 10:03 p.m., Love hits Dillon out in the front.
  Fact 0.1: Dillon makes his defender miss.
  Fact 0.2: Dillon races beyond the marker for a first down.
Sentence 1: 7:06 p.m. – Anthony Hankerson gets two first downs for the Buffs on a four-yard run and a 12-yard run.
  Fact 1.0: At 7:06 p.m., Anthony Hankerson gets two first downs for the Buffs.
  Fact 1.1: Anthony Hankerson's first downs come on a four-yard run.
  Fact 1.2: Anthony Hankerson's first downs come on a 12-yard run.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, The New York Times, University of Tennessee, Super Bowl, DeAndre Hopkins, Lions, Rams, Tyreek Hill, N.F.L. Executives on the Biggest Questions
Facing Super Bowl Contenders, Don Hutson, Wes Welker, Elroy Hirsch, Santana Moss, Raymond Berry, Jerry Rice
</item 1>

<item 2>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
It wouldn't make much sense for a four-win team to rush back its franchise player.

Patriots dual-purpose back Rhamondre Stevenson (ankle) suffered an injury on a nasty hip-drop tackle in New England's Week 13 loss to the Chargers. Bill
Belichick and company already noted that 'Mondre would miss multiple weeks, so we're bumping Ezekiel Elliott way up against the Steelers.

D'Andre Swift took a hard hit and exited Philadelphia's game late against the 49ers, but it's tough to know if he's actually hurt or was just kept out because
of the lopsided score. If Swift can't go in Week 14 against Dallas, Kenneth Gainwell and Boston Scott will split touches, with Gainwell likely garnering flex
value.

Jonathan Taylor underwent thumb surgery last week and is expected to miss a few weeks. Tough for Taylor owners to ride the home stretch of the regular season
without JT, but at least they could get him back for the end of the fantasy playoffs.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Rhamondre Stevenson
- Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game -MENTIONS-> Rhamondre Stevenson

Entities: Rhamondre Stevenson, Boston Scott, Dallas, fantasy football, Bill Belichick, Philadelphia, New England's Week 13 loss to the Chargers, Philadelphia's
game late against the 49ers, Week 14 against Dallas, Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back, Kenneth Gainwell,
Sporting News, D'Andre Swift, Chargers, Patriots
</item 2>

<item 3>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
49ers

26 Javonte Williams, Broncos vs. Browns

27 AJ Dillon, Packers @ Lions

28 Jerome Ford, Browns @ Broncos

29 Devin Singletary, Texans vs. Jaguars

30 Kareem Hunt, Browns @ Broncos

31 James Cook, Bills @ Eagles

32 Jaylen Warren, Steelers @ Bengals

33 Najee Harris, Steelers @ Bengals

34 Khalil Herbert, Bears @ Vikings

35 Tyler Allgeier, Falcons vs. Saints

36 Chuba Hubbard, Panthers @ Titans

37 Ezekiel Elliott, Patriots @ Giants

38 Latavius Murray, Bills @ Eagles

39 Roschon Johnson, Bears @ Vikings

40 Joshua Kelley, Chargers vs. Ravens

41 Keaton Mitchell, Ravens @ Chargers

42 Ty Chandler, Vikings vs. Bears

43 Dameon Pierce, Texans vs.
</item 3>

<item 4>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.

Graph facts:
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Harris
- Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant -MENTIONS-> Harris
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> Harris
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> NFL Fantasy Football
- Harris -PARTICIPATED_IN-> Harris’s remarks

Entities: NFL Fantasy Football, Pierre Strong, Harris, Week 17 win, Week 14 game, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to
know for every game, CBSSports.com, Zack Moss, Joe Mixon, Baltimore
</item 4>

<item 5>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Both Darrell Henderson and Royce Freeman (@ Cardinals) would be decent dart throws in a week filled with byes, but their floors in a zero-bye week seem too low
to trust them if Kyren Williams is back and at 100 percent. The matchup is juicy, but we have no idea what to expect from a touch-share perspective, so these
guys are no better than desperation, TD-dependent flexes.

Speaking of guys coming back from injury, we saw Khalil Herbert (@ Vikings) return to the field for Chicago against the Lions last week. However, his
performance left a lot to be desired. He finished with just 35 yards on 16 carries and wound up vastly overshadowed by Justin Field's 100-yard rushing effort.
Even worse, D'Onta Foreman scored the Chi's lone rushing TD on the afternoon.

I don't care what your team looks like — you shouldn't be starting Chuba Hubbard or Miles Sanders (@ Titans).

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Miles Sanders, Titans, Chicago, Cardinals, Royce Freeman, Lions, Sporting News, Kyren Williams, D'Onta Foreman, Chuba Hubbard, Darrell
Henderson, Justin Field, Khalil Herbert, Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back
</item 5>

<item 6>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
JACKSONVILLE JAGUARS | RB Travis Etienne: QB Trevor Lawrence hasn’t quite taken the step forward everyone thought he would, but fortunately for the Jags,
Etienne is emerging as one of the top RBs in the league. He has rushed for multiple touchdowns in three of his past four games and scored a touchdown in five of
the team’s six wins. Etienne ranks third the NFL in rushing touchdowns (seven) and sixth in rushing yards (583).

TENNESSEE TITANS | DT Jeffery Simmons: There haven’t been too many bright spots for the Titans, but Simmons continues to be a wrecking ball on the defensive
line. He leads the team in tackles for loss (six) and ranks second in sacks (4.5). If not for Simmons, Tennessee’s defense would rank a heck of a lot lower than
20th in yards allowed per game (337.8). — Michael Gallagher

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Trevor Lawrence, Jacksonville Jaguars, Tennessee Titans, Travis Etienne, Yardbarker, MVPs for all 16 AFC teams at the midpoint, Jeffery Simmons,
Michael Gallagher
</item 6>

<item 7>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
We'll talk more about Derrick Henry (concussion) in the injury section, but we can say with confidence here and now that Tyjae Spears (@ Dolphins) should be the
hottest waiver-wire pickup of Week 14. Tennessee is a run-first offense and King Henry might be sidelined with a head injury, so Spears could very well pick up
where he left off against Indy last weekend. The young back averaged 4.7 yards per carry (just shy of his 5.1 YPC entering the week) en route to 88 yards.

Another backup getting a boost this week is the Patriots' Ezekiel Elliott (@ Steelers), who will assume lead-back duties after Rhamondre Stevenson went down
with a pretty bad ankle injury in Week 13. Zeke might put in a vintage performance considering James Conner just went back to Pittsburgh and hung 105 yards and
two TDs on his old squad.

Could this be the weekend that Keaton Mitchell (vs.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Rhamondre Stevenson
- Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Rhamondre Stevenson
- Patriots vs. Broncos live score, updates, highlights from NFL Sunday night football game -MENTIONS-> Rhamondre Stevenson

Entities: Rhamondre Stevenson, Dolphins, James Conner, Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back, Sporting News,
Tennessee, Tyjae Spears, Indy, Derrick Henry, Keaton Mitchell, Pittsburgh, Patriots, Ezekiel Elliott, Steelers
</item 7>

<item 8>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Eagles

15 Kyren Williams, Rams @ Ravens

16 Joe Mixon, Bengals vs. Colts

17 Javonte Williams, Broncos @ Chargers

18 Derrick Henry, Titans @ Dolphins

19 D'Andre Swift, Eagles @ Cowboys

20 Jaylen Warren, Steelers vs. Patriots

21 Alexander Mattison, Vikings @ Raiders

22 Jerome Ford, Browns vs. Jaguars

23 Kenneth Walker III, Seahawks

24 Najee Harris, Steelers vs. Patriots

25 James Cook, Bills @ Chiefs

26 Zack Moss, Colts @ Bengals

27 Keaton Mitchell, Ravens vs. Rams

28 AJ Dillon, Packers @ Giants

29 Ezekiel Elliott, Patriots @ Steelers

30 Aaron Jones, Packers @ Giants

31 Jerick McKinnon, Chiefs @ Bills

32 Tyjae Spears, Titans @ Dolphins

33 Gus Edwards, Ravens vs. Rams

34 Dameon Pierce, Texans @ Jets

35 Roschon Johnson, Bears vs.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Raiders
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Raiders, Jaguars, Vikings, Alexander Mattison, Bills, Titans, Bengals, Packers, Texans, Dolphins, Eagles, fantasy football, PPR, Ravens, running back
</item 8>

<item 9>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Patriots

22 Jerome Ford, Browns vs. Jaguars

23 Alexander Mattison, Vikings @ Raiders

24 Najee Harris, Steelers vs. Patriots

25 Ezekiel Elliott, Patriots @ Steelers

26 Gus Edwards, Ravens vs. Rams

27 James Cook, Bills @ Chiefs

28 Zack Moss, Colts @ Bengals

29 Keaton Mitchell, Ravens vs. Rams

30 AJ Dillon, Packers @ Giants

31 Dameon Pierce, Texans @ Jets

32 Aaron Jones, Packers @ Giants

33 Clyde Edwards-Helaire, Chiefs vs. BIlls

34 Jerick McKinnon, Chiefs vs. Bills

35 Roschon Johnson, Bears vs. Lions

36 Tyjae Spears, Titans @ Dolphins

37 Kareem Hunt, Browns vs. Jaguars

38 Tyler Allgeier, Falcons vs. Buccaneers

39 Chuba Hubbard, Panthers @ Saints

40 D'Onta Foreman, Bears vs.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Raiders
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Raiders, Jaguars, Vikings, Alexander Mattison, Saints, Bills, Titans, Bengals, Packers, Texans, Dolphins, fantasy football, Ravens, running back,
Kareem Hunt
</item 9>

<item 10>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Keep rolling with Alexander Mattison (vs. Bears), too, against another below-average run defense.

Welcome to the top 25, Zach Charbonnet (vs. 49ers)! With Kenneth Walker III's oblique injury comes opportunity for the rookie, who already looked strong in the
month leading up to Seattle's narrow Week 11 loss to the Rams. Sure, Charbs' 69-yard effort in Los Angeles last week was a bit underwhelming, but we love his
upside with a full game as the Seahawks' bell-cow.

Speaking of bell-cow backs, things are looking very promising for the short-term workload of Green Bay's AJ Dillon (@ Lions). Veteran Aaron Jones (knee) and
rookie Emanuel Wilson (shoulder) both got hurt in Week 11, so Dillon should be the main man in Motown for the Pack.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Alexander Mattison
- Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Alexander Mattison
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Alexander Mattison
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Alexander Mattison
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Alexander Mattison

Entities: Alexander Mattison, Green Bay Packers, Lions, Sporting News, Rams, Seattle Seahawks, Kenneth Walker III, Aaron Jones, Los Angeles, Zach Charbonnet, AJ
Dillon, Emanuel Wilson, Motown, Seattle's narrow Week 11 loss to the Rams, Bears
</item 10>
=== END ITEM id=04ae99cd6857d424 ===
