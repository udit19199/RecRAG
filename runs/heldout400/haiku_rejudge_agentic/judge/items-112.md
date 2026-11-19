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

=== ITEM id=1e5fe1a6204222e0 ===
Gold sentences and facts:
Sentence 0: But by the age of 16, she is public property – incapable of going outside without being mobbed, and her love of singing and dancing now a lucrative resource.
  Fact 0.0: By the age of 16, the person described is public property.
  Fact 0.1: By the age of 16, the person described is incapable of going outside without being mobbed.
  Fact 0.2: By the age of 16, the person described's love of singing and dancing is now a lucrative resource.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker described it as almost a very strange, subtle clue to the media.
  Fact 1.1: The clue signals that the media does not know everything that happened in that relationship.
  Fact 1.2: The speaker says something really major and traumatic could have happened to her that the media does not know about.
  Fact 1.3: The speaker told NPR this.

Retrieved items:
<item 1>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
There is no love lost for her family

Britney writes about how she was always made to feel she was not enough as a kid; her father’s alleged alcoholism, neglect and brutal standards; how her mother
always made her feel less-than. She is unsparing about her younger sister, who she characterises as a brat who was spared the pain of Britney’s impoverished
childhood because of her success. “My mom and I had to witness the ugliness and the violence without believing that there was anywhere else to go.”

Even once Britney is the biggest pop star in the world, her parents still seem to have an undue say in the running of her career, including making her sit for
humiliating TV interviews – why exactly that is remains unanswered. Her father seizes control legally with the imposition of the conservatorship, telling her,
chillingly, “I am Britney Spears now,” and her family appear to live off her dime while she lives an extraordinarily restricted life.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, conservatorship, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 1>

<item 2>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She has never really known freedom

As a child, Britney escaped her difficult home life – her alcoholic father often rowing with her mother – into the outdoors, which “gave me a sense of aliveness
and danger”, she writes. Performing also makes her feel invincible. But by the age of 16, she is public property – incapable of going outside without being
mobbed, and her love of singing and dancing now a lucrative resource.

By the time of her second album, she wants for so little: skinny-dipping with her dancers after playing Rock in Rio in January 2001 is one of her happiest
memories. “It was a taste of rebellion and freedom, but I was just having fun and being a 19-year-old,” she writes.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, Rock in Rio, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 2>

<item 3>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
At what point did I promise to stay 17 for the rest of my life?”

What she knows now, she writes, “is that every part of normal life had been stripped from me – going out in public without becoming a headline, making normal
mistakes as a new mother of two babies, feeling like I could trust the people around me. I had no freedom and yet also no security. At the time I was also
suffering, I now know, from severe postpartum depression.” It left her suicidal. “I’ll admit it, I felt that I couldn’t live if things didn’t get better.”

She knows she is trapped in a state of arrested development

It’s often said that famous people are frozen in time at the age they became famous, but Britney was never allowed to be a real child, nor a real adult. Her
2001 single I’m Not a Girl, Not Yet a Woman may be her most prescient recording; she is painfully conscious of this divide.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, I’m Not a Girl, Not Yet a Woman, postpartum depression, arrested development, ‘I was being ritually tortured’: the most shocking
moments in Britney Spears’ memoir, The Guardian
</item 3>

<item 4>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Justin Timberlake, she dreams about quitting pop and opening a shop: “With the gift of hindsight I can see that I hadn’t given myself
enough time to heal from the breakup,” she writes – though her requests for a break are not only turned down, but interpreted as a potential sign of sedition to
be quashed and interrogated.

In this restrictive environment, small kindnesses take on an outsized significance, like Paris Hilton encouraging her “to have fun for the first time in a long
time”, leading to some infamous nights out with the heiress and Lindsay Lohan. Once again, Britney is seen as stepping out of bounds, potentially damaging her
earning potential. One night she has her mother babysit her sons; when she returns home tipsy, Lynne Spears screams at her, she writes. “The shame I felt killed
my heart. I stood there, reeling, and thought, OK, I guess it’s forbidden for me to party.”
</item 4>

<item 5>
Graph query row:
publisher: The Guardian
title: ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
passage: ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She has never really known freedom

As a child, Britney escaped her difficult home life – her alcoholic father often rowing with her mother – into the outdoors, which “gave me a sense of aliveness
and danger”, she writes. Performing also makes her feel invincible. But by the age of 16, she is public property – incapable of going outside without being
mobbed, and her love of singing and dancing now a lucrative resource.

By the time of her second album, she wants for so little: skinny-dipping with her dancers after playing Rock in Rio in January 2001 is one of her happiest
memories. “It was a taste of rebellion and freedom, but I was just having fun and being a 19-year-old,” she writes.
connected_entity: Justin Timberlake
</item 5>

<item 6>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Timberlake, she likens herself to Benjamin Button. “Somehow that year, in becoming more vulnerable I started to feel like a child again.”

The situation worsens amid postnatal depression. “It was as if some part of me became the baby,” she writes in an apologetic section about transferring her
frustrations on to her interior decorator. “One part of me was a very demanding grown woman yelling about white marble, while another part of me was suddenly
very childlike.”

When she is placed under the conservatorship, she not only regresses, but feels dehumanised by the constant scrutiny. “I became a robot. But not just a robot –
a sort of child-robot. The conservatorship stripped me of my womanhood, made me into a child.”

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, conservatorship, Britney Spears’ memoir, postnatal depression, Timberlake, Benjamin Button, interior decorator, ‘I was being ritually
tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 6>

<item 7>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
It’s posting selfies – either nude or clothed – on Instagram. Some people don’t understand the latter, she writes, “but I think if they’d been photographed by
other people thousands of times, prodded and posed for other people’s approval, they’d understand that I get a lot of joy from posing the way I feel sexy and
taking my own picture, doing whatever I want with it.”

Her ‘virginity’ was a prison

Britney was actively marketed as a virgin, a repulsive, archaic premise that made the 16-year-old’s inevitable sexuality into a kind of timebomb. In fact, she
writes happily, she first slept with her older brother’s best friend when she was 14. Yet the fixation on her “purity” took “the focus off me as a musician and
a performer”, she writes. “All some reporters could think of to ask me was whether or not my breasts were real (they were, actually) and whether or not my hymen
was intact.”

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, Britney Spears’ memoir, virginity, sexuality, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir,
The Guardian, Instagram
</item 7>

<item 8>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
As the restrictions get tighter, Britney acts out – and ends up legally denied her freedom for 13 years, under a conservatorship controlled by her father.

Her body felt like public property

From day one, Britney’s body is made fair game. Between the videos for her debut single, Baby, One More Time and Sometimes, the media noted her changing
physique and concluded that the 16-year-old must have had breast implants – rather than simply going through puberty. She notices that talkshow hosts ask her
about her chest, while her boyfriend Justin Timberlake gets asked serious questions about his music. It’s one of many pressures that ultimately push her to
breaking point, and when she writes about shaving her head in a salon in 2007, she frames it as a “fuck you” to the impossible beauty standards she felt forced
to live up to: “I’d smiled politely while TV show hosts leered at my breasts, while American parents said I was destroying their children by wearing a crop
top.”

Graph facts:
- Justin Timberlake -PARTICIPATED_IN-> Britney Spears and Justin Timberlake breakup
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Justin Timberlake
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Justin Timberlake
- Justin Timberlake -PARTICIPATED_IN-> American Music Awards
- Justin Timberlake -PART_OF-> NSYNC

Entities: Justin Timberlake, Britney Spears, conservatorship, Baby, One More Time, Sometimes, impossible beauty standards, shaving her head, ‘I was being
ritually tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 8>

<item 9>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
The conservatorship stripped me of my womanhood, made me into a child.” It’s hard to explain, she says, “how quickly I could vacillate between being a little
girl and being a teenager and being a woman, because of the way they had robbed me of my freedom … They wanted me to be wild onstage, the way they told me to
be, and to be a robot the rest of the time.”

Music seems to have become an afterthought

Early on, Britney writes, she loved making music. She told the producer Max Martin that she wanted to sound more R&B than straight pop; she listened to Tainted
Love the night before recording Baby, One More Time and stayed up late so that she would sound as gravelly and “fried” as Marc Almond. She was a studio
perfectionist who would stay in the booth for hours, and the school setting and uniforms of the Baby video were her idea.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, Tainted Love, Baby, One More Time, Marc Almond, R&B, straight pop, Baby video, ‘I was being ritually tortured’: the most shocking
moments in Britney Spears’ memoir, Max Martin, The Guardian, music
</item 9>

<item 10>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
When she arrives, so does another Swat team. “I’m a five-foot four-inch pop singer who calls everyone sir and ma’am,” she writes. “They treated me like I was a
criminal or predator.”

The conservatorship clamps down on her life, ruled by her father, who she alleges was “an alcoholic, someone who’d declared bankruptcy, who’d failed in
business, who’d terrified me as a little girl”. She is medicated and surveilled. Her dates are required to agree to background checks and blood tests, made to
sign NDAs, and receive a full rundown of her sexual history before the first date. “The insanity of this system kept me from finding basic companionship, having
a fun night out or making new friends – let alone falling in love,” Britney writes. “It was just the worst thing that could possibly ever happen to my music, my
career and my sanity.”

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, conservatorship, Swat team, her father, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir, The
Guardian
</item 10>
=== END ITEM id=1e5fe1a6204222e0 ===

=== ITEM id=d7e126feffd85f17 ===
Gold sentences and facts:
Sentence 0: The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury.
  Fact 0.0: The winger came firing out of the gate.
  Fact 0.1: The outlook has changed for the Devils in recent days.
  Fact 0.2: The Devils lost the superstar center Hughes to injury.
  Fact 0.3: The outlook for the Devils changed after they lost Hughes to injury.
Sentence 1: Having to depend much on his weaker arm, Hamlin stayed in contention for the championship four race until the final race in Martinsville.
  Fact 1.0: Hamlin had to depend much on his weaker arm.
  Fact 1.1: Hamlin stayed in contention for the championship four race.
  Fact 1.2: Hamlin stayed in contention until the final race in Martinsville.

Retrieved items:
<item 1>
Graph query row:
article: Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche
published_on: 2023-11-07T22:09:00+00:00
publisher: Sporting News
relationship: MENTIONS
connected_entity: NHL
page: Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
passage: Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Tuesday night's NHL national games feature an Original Six matchup at Madison Square Garden in addition to a battle between the team that left and the team that
took its place. The Rangers and Red Wings will face off at the world's most famous arena as both squads have jumped out to hot starts in the new season. Dylan
Larkin is a game-time decision for the Motor City while injuries to the Blueshirts' Adam Fox, Filip Chytil and Igor Shesterkin are making sure some of this
matchup's biggest stars will not be watching in uniform. Regardless, there is still plenty of star power on Broadway in what should feature plenty of offense.

In the late game, the Devils return to Colorado -- the place they left in order to move to New Jersey -- for a showdown with the Avalanche. Again, injuries are
a big part of the storyline here, thanks to Jack Hughes and Nico Hischier out until further notice.
</item 1>

<item 2>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
After running the car full-time in 2023, it was announced Thursday that A.J. Allmendinger would go back to full-time Xfinity Series racing for Kaulig in 2024,
though he will continue to run a partial Cup schedule.

The assumption of many had been that the No. 16 would be taken over by Ty Dillon, but that assumption may have been premature. Weaver shared in a recent thread
on X that information suggesting Dillon had been hired to drive the No. 16 was "incorrect," casting ambiguity on both who will join Daniel Hemric at Kaulig next
season as well as the exact plans of Dillon, who has bounced from team to team since Germain Racing shut down following the 2020 season.

Nuts and Bolts
</item 2>

<item 3>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
USA Today via Reuters Jul 30, 2023; Richmond, Virginia, USA; NASCAR Cup Series driver Chris Buescher (17) celebrates in Victory Lane after winning the Cook Out
400 at Richmond Raceway. Mandatory Credit: Amber Searls-USA TODAY Sports

Overall, he has been one of the season’s standout performers and displayed incredible grit and consistency throughout the year. His performances in 2023 make a
strong case for a place in the best 50 drivers of the year.

2. Denny Hamlin

Finding a fan of Denny Hamlin in the NASCAR fandom is a tough task. Continuously surrounded by controversy, Joe Gibbs Racing‘s #11 Toyota Camry driver has been
subject to a lot of criticism. However, his ability as a race car driver has never been in doubt. As the season entered its decisive phase of the playoffs, it
seemed like 2023 would be the year Hamlin’s championship drought finally ended.

Graph facts:
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -MENTIONS-> NASCAR
- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change -MENTIONS-> NASCAR
- NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 -MENTIONS-> NASCAR

Entities: NASCAR, Chris Buescher, Cook Out 400, Richmond Raceway, Richmond, Virginia, USA, Amber Searls, Denny Hamlin, Toyota Camry, Top 50 Motorsport Drivers
in 2023, Autosport, Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023, NASCAR Cup Series, Essentially Sports, Reuters, playoffs
</item 3>

<item 4>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Autosport has revealed its annual list of the 2023’s Top 50 drivers. The renowned motorsports publishing brand has continued the tradition since 2002, and every
year, the announcement creates some suspense within the racing community. Highlighting the 50 standout performers in motorsports for the year, Autosport has
included several NASCAR drivers in the prestigious list.

America’s Favorite Video Today

William Byron leads the list of NASCAR drivers with a 12th-place ranking as Kyle Larson and 2023 Cup Series champion Ryan Blaney join him. The scarcity of
NASCAR drivers in the rankings has raised some eyebrows. With the year producing numerous cases of NASCAR drivers standing out with their consistency and skills
behind the steering wheel, let us look at some of the major exclusions from Autosport’s Top 50 Drivers of 2023.

1. Chris Buescher

ADVERTISEMENT Article continues below this ad

RFK Racing found a fresh breath of motivation this year.

Graph facts:
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -MENTIONS-> NASCAR
- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change -MENTIONS-> NASCAR
- NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 -MENTIONS-> NASCAR

Entities: NASCAR, Cup Series, RFK Racing, Chris Buescher, Autosport, Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023, Essentially
Sports, William Byron, motorsports, Ryan Blaney, Kyle Larson
</item 4>

<item 5>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
3. Christopher Bell

As unfortunate as Denny Hamlin was, his JGR teammate Christopher Bell must have felt hard done by after an impressive season. The #20 Toyota Camry driver
progressed to the championship race for a second successive season. However, a cruel twist of fate waited for him in the season finale at Phoenix Raceway, as he
was forced to retire early from the biggest race of his career with a broken brake rotor.

via Imago NASCAR, Motorsport, USA Food City Dirt Race Apr 9, 2023 Bristol, Tennessee, USA NASCAR Cup Series driver Christopher Bell 20 wins the Food City Dirt
Race at the Bristol Motor Speedway Dirt Course. Bristol Bristol Motor Speedway Dirt Course Tennessee USA, EDITORIAL USE ONLY PUBLICATIONxINxGERxSUIxAUTxONLY
Copyright: xRandyxSartinx 20230409_lbm_bs1_024

Bell’s 2023 season has gone under the radar.

Graph facts:
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -MENTIONS-> NASCAR
- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change -MENTIONS-> NASCAR
- NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 -MENTIONS-> NASCAR

Entities: NASCAR, Denny Hamlin, Toyota Camry, JGR, Food City Dirt Race, Bristol Motor Speedway Dirt Course, Bristol, Tennessee, Motorsport, broken brake rotor,
Christopher Bell, NASCAR Cup Series, Autosport, Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023, Essentially Sports
</item 5>

<item 6>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Mandatory Credit: Gary A. Vasquez-USA TODAY Sports

While his on-track expertise was one of the notable points in 2023, his off-track antics didn’t go unnoticed either. Ross Chastain was clouded in controversy
after a fistfight with Noah Gragson and on-track altercations with Denny Hamlin and Kyle Larson. While his cutthroat racing style appeased the fans, his driving
skills didn’t waver either.

ADVERTISEMENT Article continues below this ad

Chastain scored two wins, ten 1op-5s, and 14 top-tens, while leading 640 laps in 2023. He also led the NASCAR Cup Series standings for seven weeks and had a
career-best 28 lead lap finishes. Overall, Ross Chastain has been one of the most resilient NASCAR drivers of 2023 and deserves a mention among NASCAR’s very
best.

Graph facts:
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -MENTIONS-> NASCAR
- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change -MENTIONS-> NASCAR
- NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 -MENTIONS-> NASCAR

Entities: NASCAR, Denny Hamlin, NASCAR Cup Series, Autosport, Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023, Ross Chastain,
Essentially Sports, fistfight, on-track altercations, Kyle Larson, Noah Gragson
</item 6>

<item 7>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
The 43-year-old dominated the first round of the playoffs and was the driver to watch out for. He eventually won his first playoff race in Bristol and cruised
to the next round. However, Hamlin revealed recently that a shoulder injury prevented him from being at the best of his abilities as the playoffs progressed.
Having to depend much on his weaker arm, Hamlin stayed in contention for the championship four race until the final race in Martinsville.

He secured three wins, 14 top-fives, and 19 top-10 finishes in 2023. He also led a whopping 998 laps, which makes his omission from Autosport’s list seem
unjustified. He finished 5th in the final standings and was one of the most consistent and best performers of the 2023 NASCAR season. From the looks of it, a
championship trophy would have surely put him in the upper echelons of Autosport’s Top 50.

3.

Graph facts:
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -MENTIONS-> NASCAR
- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change -MENTIONS-> NASCAR
- NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 -MENTIONS-> NASCAR

Entities: NASCAR, championship four race, Hamlin, Martinsville, first playoff race, final race in Martinsville, Bristol, Autosport, Major NASCAR Snubs as
Autosport Releases the Top 50 Motorsport Drivers in 2023, Essentially Sports, 2023 NASCAR season, shoulder injury
</item 7>

<item 8>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Since Brad Keselowski arrived in 2022, the team has continued evolving on and off the track. While Keselowski has certainly instilled his experience and winning
mentality within the team, he failed to produce a race win in 2023. Instead, it was Chris Buescher who outshone the Cup Series veteran with his best-ever season
in NASCAR’s premier competition.

Chris Buescher ended his season with three wins, nine top-5s, and 17 top-tens. Two of his three wins in 2023 came in succession at Richmond and Michigan.
Buescher continued his dream run in the Cup Series playoffs and eventually finished 7th in the final standings. The #17 Ford Mustang driver could not, however,
secure a win in the Round of 8 and did not have enough points to make a case for the championship four race at Phoenix.

Graph facts:
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -MENTIONS-> NASCAR
- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change -MENTIONS-> NASCAR
- NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 -MENTIONS-> NASCAR

Entities: NASCAR, Cup Series, Brad Keselowski, Chris Buescher, Richmond, Cup Series playoffs, Round of 8, championship four race, 2023 race win at Richmond,
2023 race win at Michigan, Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023, Essentially Sports, Michigan, Ford Mustang, Phoenix
</item 8>

<item 9>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Two more drivers to have featured in the 2023 NASCAR season and Autosport’s Top 50 Drivers are Supercars racers Shane van Gisbergen and Brodie Kostecki. While
the pair battled for the championship in the Australian racing circuit, they also raced at the Indianapolis road course race earlier this season. SVG had
earlier created history by winning in his NASCAR debut at the inaugural Chicago Street Race.

ADVERTISEMENT Article continues below this ad

These are some major exclusions that could have been featured in Autosport’s annual list. At the same time, motorsports is a vast field with elite competitions
taking place the world over. Some fan favorites are bound to be left out.

Graph facts:
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil -MENTIONS-> NASCAR
- “We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change -MENTIONS-> NASCAR
- NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 -MENTIONS-> NASCAR

Entities: NASCAR, Autosport, Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023, Essentially Sports, Shane van Gisbergen, Brodie
Kostecki, Supercars, 2023 NASCAR season, championship, Indianapolis road course race, Chicago Street Race, Australian racing circuit, Indianapolis, motorsports
</item 9>

<item 10>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
The team had two drivers in the Cup Series playoffs for the first time since its inception. One of the major reasons behind its success in 2023 is #45 Toyota
Camry driver Tyler Reddick.

Alongside team owner Denny Hamlin, Tyler Reddick was the pick of the drivers in the first round of the NASCAR playoffs. The 27-year-old California native
showcased his road course prowess with a win at COTA earlier this year. Carrying the momentum to the playoffs, Reddick won in Kansas in the Round of 16 to gain
more traction for his championship push. He also had one of the best average starting positions among the drivers, with an average of 10.167, despite having
only two poles to his name.

Watch This Story: NASCAR legend Kevin Harvick’s unexpected EV confession

Tyler Reddick finished sixth in the final standings and ended the season with two wins, ten top-5s, and 16 top-10s.

Graph facts:
- The people who ruined the internet -MENTIONS-> Kansas
- The best tabletop RPGs we played in 2023 -MENTIONS-> Kansas
- John Heard -LOCATED_IN-> Kansas
- NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal -MENTIONS-> NASCAR
- Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash -MENTIONS-> NASCAR

Entities: Kansas, NASCAR, California, Denny Hamlin, Toyota Camry, Autosport, Tyler Reddick, Kevin Harvick, COTA, Cup Series, NASCAR playoffs, COTA win, Kansas
win, Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023, Essentially Sports
</item 10>
=== END ITEM id=d7e126feffd85f17 ===

=== ITEM id=de84cf924e03d5d8 ===
Gold sentences and facts:
Sentence 0: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 0.0: The second week of the trial had standout testimony.
  Fact 0.1: The standout testimony of the second week came from Alameda Research's former CEO Caroline Ellison.
  Fact 0.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 0.3: Caroline Ellison claimed she used the $14 billion to repay debts to lenders.
  Fact 0.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 0.5: Caroline Ellison claimed she did this under the instruction of SBF.
Sentence 1: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 1.0: SBF is the name by which he is known.
  Fact 1.1: SBF has pleaded not guilty to some seven charges of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The full breakdown of his testimony on his relationship with FTX, stretching back to 2019, can be found right here.

We also got a look at the composition of the SBF trial’s jury, which ranges from a train conductor to a retired investment banker who attended Stanford
University. You can listen to a special joint episode of our podcasts Equity and Chain Reaction to get a better sense of the vibe between the jurors and the
rest of the court.

For a full rundown of our coverage of the SBF trial, check below:

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Josh Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Jacquelyn Melinek
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Chain Reaction podcast
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Apple Podcasts

Entities: SBF Trial: The latest updates from the FTX collapse’s courtroom drama, SBF Trial, FTX collapse, SBF, Chain Reaction, SBF trial’s jury, TechCrunch,
Equity, FTX, Stanford University
</item 1>

<item 2>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 2>

<item 3>
Graph query row:
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
published_on: 2023-10-06T17:22:02+00:00
publisher: TechCrunch
person: Caroline Ellison
relationship: MENTIONS
statement: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 3>

<item 4>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
You can also follow along with her and the TechCrunch team’s coverage in the Chain Reaction newsletter, which drops Thursdays at 12 p.m. PT, and the Chain
Reaction podcast, which will hit your feed every Thursday.

How to follow the SBF trial

The trial kicked off with jury selection October 3. From there, access to the proceedings is limited, as no devices are allowed; reports are being physically
put together within the courtroom (or sent to overflow rooms).

Beyond our daily coverage, additional expertise and commentary will live on TechCrunch+. There, you can find stories like this in-depth breakdown of what to
expect from both sides of the SBF case, where the prosecution and defense could gain ground or fall short in their arguments and what the takeaways were from
the opening arguments.

SBF trial: What we learned in week three

The third week of the trial featured a couple of noteworthy moments for the prosecution, including FTX’s former general counsel Can Sun.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Jacquelyn Melinek
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Apple Podcasts
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Josh Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> bankruptcy

Entities: Chain Reaction podcast, Chain Reaction newsletter, jury selection, SBF Trial: The latest updates from the FTX collapse’s courtroom drama, SBF Trial,
TechCrunch, TechCrunch+, Can Sun, FTX
</item 4>

<item 5>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Chain Reaction podcast
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Apple Podcasts
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Jacquelyn Melinek
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> bankruptcy

Entities: Josh Naftalis, SBF Trial: The latest updates from the FTX collapse’s courtroom drama, SBF Trial, Alameda, TechCrunch, money laundering, FTX, Sam
Bankman-Fried, fraud
</item 5>

<item 6>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> FTX collapse
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> SBF’s trial
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> The New York Times
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> SBF
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Series C round

Entities: FTX collapse, SBF’s trial starts soon, but how did he — and FTX — get here?, Damian Williams, Gary Wang, Caroline Ellison, Bankman-Fried’s trial, Wang
and Ellison pleaded guilty to federal criminal charges, Bankman-Fried pleaded not guilty, financial fraud, conspiracy to commit money laundering, conspiracy to
misuse customer funds, civil penalties, federal criminal charges, Alameda Research, TechCrunch
</item 6>

<item 7>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
He, like many of SBF’s former colleagues, opted to cooperate with authorities and testified that SBF asked him to create a theoretical “legal justification” for
the use of billions in consumer funds just days before FTX filed for bankruptcy.

Robert Boroujerdi, managing director of the hedge fund Third Point, testified about his experience working with SBF and FTX, in a partnership that ultimately
resulted in a $60 million investment and a lack of awareness about the special relationship between Alameda Research and FTX. Had he known of such a
relationship, he testified that Third Point would not have invested in FTX in the first place.

And earlier in the week, a University of Notre Dame professor who helped the government prosecution of Enron and WorldCom, Peter Easton, testified about what he
uncovered after tracing the flow of billions of dollars between Alameda Research and FTX.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Jacquelyn Melinek
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Chain Reaction podcast
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Josh Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Apple Podcasts

Entities: Robert Boroujerdi, Third Point, University of Notre Dame, SBF Trial: The latest updates from the FTX collapse’s courtroom drama, SBF Trial, SBF,
Alameda Research, TechCrunch, Peter Easton, FTX
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Ellison
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Ellison
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Ellison

Entities: Ellison, New York, Jane Rosenberg, FTX, U.S. attorney's office, courthouse, pleading guilty, Wang's testimony, Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court, Wang, hearing, Sam Bankman-Fried, Reuters, Manhattan, CNBC
</item 8>

<item 9>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF alongside rich and powerful figures to illustrate
the peak of his power and influence prior to FTX’s implosion.

“This man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he
emptied the accounts of ordinary customers, too. He bought himself wealth, power and influence.”

The defense, on the other hand, depicted SBF and his colleagues as in over their heads, without ill-will as FTX collapsed around them.

“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said. “They had to figure out how to navigate a world where
they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of their customers, managing hundreds of employees, all
while building up their actual exchange.”

Read a recap of the prosecution and defense’s full arguments here.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Josh Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Jacquelyn Melinek
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Chain Reaction podcast
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Apple Podcasts

Entities: SBF Trial: The latest updates from the FTX collapse’s courtroom drama, SBF Trial, FTX collapse, SBF, Thane Rehn, Mark Cohen, TechCrunch, FTX
</item 9>

<item 10>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Much of said funding came from customers, and Easton testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make
investments or direct funding toward political causes and charities.

SBF trial: What we learned in week two

The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to
repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also went into detail about how a $2 billion venture fund, FTX
Ventures, was funded by money that had previously been allocated to Alameda from third-party lenders.

And though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1 billion in
frozen trading accounts.

SBF trial: What we learned in the week one

As expected, the opening statements in the SBF trial were spicy.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Josh Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Naftalis
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Jacquelyn Melinek
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Chain Reaction podcast
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Apple Podcasts

Entities: SBF Trial: The latest updates from the FTX collapse’s courtroom drama, Easton, FTX Ventures, Chinese officials, Paying Chinese officials $150 million
to restore access to $1 billion in frozen trading accounts, charities, SBF Trial, SBF, Caroline Ellison, Alameda Research, bribery, TechCrunch, political
causes, real estate, FTX
</item 10>
=== END ITEM id=de84cf924e03d5d8 ===
