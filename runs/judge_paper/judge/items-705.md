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

=== ITEM id=c54645abf4c355d9 ===
Gold sentences and facts:
Sentence 0: “But nah, it was awesome.” On Saturday, the competition also said goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season, and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.
  Fact 0.0: A speaker said 'But nah, it was awesome.'
  Fact 0.1: On Saturday, the competition said goodbye to a range of players.
  Fact 0.2: On Saturday, the competition said goodbye to Western Bulldogs forward Richelle Cranston.
  Fact 0.3: Richelle Cranston has battled stage five chronic kidney disease while playing this season.
  Fact 0.4: On Saturday, the competition said goodbye to Hawthorn's Akec Makur Chuot.
  Fact 0.5: Akec Makur Chuot is a Hawthorn player.
  Fact 0.6: Akec Makur Chuot played 40 games for three clubs.
Sentence 1: The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year.
  Fact 1.0: The Dogs are Richelle Cranston's third club.
  Fact 1.1: Richelle Cranston started at Melbourne.
  Fact 1.2: Richelle Cranston had a stint with Geelong.
  Fact 1.3: Richelle Cranston joined the Bulldogs last year.

Retrieved items:
<item 1>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Tackled above waist and below shoulders. No sling or double motion. Sheppard lands on her back. ′ ‘Girls’ play hard too,” Harford said on X, the social media
site formerly known as Twitter. This echoed Collingwood defender and Fox Footy commentator Ruby Schleicher’s thoughts at the time, who called it a “monster
tackle” but believed it was “fair”. “I don’t see any dirtiness in that,” Schleicher said on Fox Footy. “It was pretty heavy on her head, wasn’t it? I look at
that – I still see a fair tackle. There’s no doubt the match review [officer] will look at it because they have to I guess. The duty of care for the player.”

The decision a week comes after Adelaide premiership defender Najwa Allen was handed a three-ban suspension, the longest ban in AFLW history, for a high bump on
Western Bulldog Kirsten McLeod. The Crows elected not to challenge the decision.
</item 1>

<item 2>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 2>

<item 3>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Michael Prior guided West Coast to a gutsy win. Credit: Getty

Aisling McCarthy and Amy Franklin drew first blood for West Coast within the opening five minutes, while the hosts were kept goalless. The Bombers only added
three behinds in the second term and the Eagles failed to score as the blustery conditions took their toll on the game. West Coast’s Kellie Gibson broke the
goal drought with the only major of the third quarter, before the home side came to life with the wind at their backs in the final term, with skipper Bonnie
Toogood and Amber Clarke reducing the margin to four points. Toogood continued to pepper the scoreboard, but the Eagles held on to secure a rare victory. “We’ve
had a horror week back at home, and it’s been really bloody tough on all of us, and Mick as well. I’m just really proud of the girls,” West Coast captain Emma
Swanson said post-game on the broadcast. “I can name the whole team.
</item 3>

<item 4>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
J’Noemi Anderson is tackled by a group of Dockers. Credit: AFL Photos The Dockers dominated the Saints from the start to win 6.9 (45) to 3.3 (21) at Fremantle
Oval. Having been on the verge of making their first finals appearance, St Kilda (4-4) could now miss the top eight even if they beat the fourth-placed Brisbane
Lions and the ninth-placed Carlton.

Dana East was everywhere for Fremantle, leading the game in disposals (19) and contested possessions (11), alongside Orlagh Lally (19 disposals, five
clearances). Jamiee Lambert was the shining light for the otherwise subdued Saints, collecting a game-high seven clearances and 13 contested possessions. Docker
Hayley Miller hunts a loose ball against St Kilda. Credit: Getty Fremantle flew out of the gates from the first siren, producing their highest-scoring quarter
of the season with 26 points in the opening term.
</item 4>

<item 5>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Geelong coach Daniel Lowther said the strong winds made it hard for his side to clear the ball out of their defensive 50-metre zone and to move the ball with
any certainty around the ground.

Loading “At one point the rain hit our box, it was flowing,” Lowther said. “Both teams had to fight in the conditions, but it didn’t help the game. “Our marking
contest wasn’t too flash, our ground ball representation was a bit hit and miss, so our defence were under a bit more pressure than they should have been, but
they fought back really well,” Lowther said. Collingwood next play a Sydney line-up including former Magpies star Chloe Molloy, while Geelong will take on
Richmond at Ikon Park.

Dockers spoil Saints’ finals hopes Fremantle recorded a comfortable 24-point win yesterday to bring an abrupt end to St Kilda’s four-game streak and dash their
AFLW finals hopes. J’Noemi Anderson is tackled by a group of Dockers.
</item 5>

<item 6>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Conversely, the score was the most the Saints have conceded in any quarter this season.

Aine Tighe set the tone for the Dockers with a brilliant goal less than 30 seconds into the first quarter, followed by two majors from Gabby O’Sullivan and
another by Angelique Stannett. Ella Friend slotted a much-needed goal for the visitors to make it 4.2 (26) to 1.3 (9) at quarter-time. With less than a minute
till the long break, captain Hayley Miller (two goals) kicked the only major of the second quarter to give Fremantle a comfortable 26-point lead at half-time.
Despite two goals from Tyanna Smith and J’Noemi Anderson in the second half, the Saints struggled to eat into the deficit. Fremantle (4-4) face reigning
premiers Melbourne next, before finishing the season against 11th-placed Sydney.
</item 6>

<item 7>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“It was an awesome game and four quarters of Port Adelaide footy … It was awesome to finish up like that,” Phillips told Fox Footy after the 13.10 (88) to 4.6
(30) win over the Giants. Erin Phillips celebrates a goal during her final AFLW match. Credit: AFL Photos Earlier, she told the ABC that she wanted her legacy
to include doing everything she could to grow the game.

“Hopefully I can be remembered as someone who took their opportunities, loved the game and was willing to do whatever it took to ensure the next generation was
in a better position,” she said. Richelle Cranston of the Bulldogs is chaired off the ground by teammates Isabelle Pritchard and Ellie Blackburn after
announcing her retirement. Credit: Getty The 38-year-old kicked a momentous goal from a set shot in the final quarter, and she was smothered by her jubilant
teammates afterwards.
</item 7>

<item 8>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I’m excited like I think the group’s coming together really nicely and where really well-prepared compared to previous years,” Riddell told Fox Footy. “So, I
can’t wait I think it’s an exciting time for us.” The game was one-way traffic against the Dogs with the game played almost entirely in the Roos’ attacking
half. Riddell credited her side’s “team performance” after two losses to the Dees and Crows. North Melbourne have secured a top-four spot for finals. Credit:
Getty For the Western Bulldogs, it was another tough day out as they notched their ninth loss of the season. They won just one game this year, an eight-point
victory last weekend over 17th-placed West Coast.

Britney Gutknecht kicked the Dogs’ only goal against the Kangaroos, which came early in the final term to avoid a record for the lowest AFLW score in the
history of the competition.
</item 8>

<item 9>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Bridie Hipwell kicked the only goal of the third quarter to give Hawthorn a game-high nine-point lead. Loading Richmond rebounded in the last term, with
Courtney Jones ending Hawthorn’s three-goal run with her first touch of the night, cutting the margin to three points. Dogged by inaccuracy in front of goal,
Yassir reduced the deficit to a single point before finally breaking through on her third attempt. That put the Tigers ahead by six points with 11 minutes left,
before Caitlin Greiser sealed the win with a massive goal from 45 metres.
</item 9>

<item 10>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Dogs forward reveals battle with kidney disease Western Bulldogs forward Richelle Cranston has revealed she’s been living, and playing, with stage five chronic
kidney disease. “I still manage to do the things I love,” the inaugural AFLW player posted to Instagram. “It’s just really hard sometimes.” Richelle Cranston of
the Bulldogs: “It’s just really hard sometimes.” Credit: Getty Images The 34-year-old posted the caption along with a video detailing her life with the disease
including weekly blood tests and daily blood pressure checks, tracking water and food intake, taking medications including resonium (to lower potassium levels),
all while training in the high-performance environment.

The Dogs are Cranston’s third club after starting at Melbourne and having a stint with Geelong before heading to the Bulldogs last year. Cranston lined up for
the Bulldogs in her side’s 57-point loss to Sydney on Saturday at Whitten Oval, 14.12 (96) to 6.3 (39).
</item 10>

<item 11>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“I don’t think I’ve been as nervous to kick a goal in my life, to be honest,” she joked. “But nah, it was awesome.” On Saturday, the competition also said
goodbye to a range of players including Western Bulldogs forward Richelle Cranston, who has battled stage five chronic kidney disease while playing this season,
and Hawthorn’s Akec Makur Chuot, who played 40 games for three clubs.

Cranston, who was an inaugural AFLW player, will start kidney dialysis in January so had no option but to hang up the boots and focus on her treatment and
health. “[I’m feeling] sad. Emotions hit me when the siren went. It’s been a really emotional week, the girls have really gotten around me. But yeah, my health
has sort of deteriorated, it’s been a really hard year,” Cranston told Fox Footy after the Bulldogs lost to North Melbourne 7.13 (55) to 1.3 (9).
</item 11>

<item 12>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 12>

<item 13>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“Richmond were just absolute beasts, especially their midfielders. They really worked hard all night,” Hawthorn coach Bec Goddard said after the game. “Our work
rate really dropped off and we just watched them play some really good footy. “If you’re a Hawks fan, you would’ve seen some really exciting moments of footy
tonight.

“Keep getting excited. We are coming, we’re growing, we’re getting better.” Hawthorn’s Irish ace Aine McDonagh kicked the opening goal before Tigers skipper
Katie Brennan slotted two unanswered majors to celebrate her 50th game in style. Fellow 50-gamer Greta Bodey put boot to ball to snatch her own goal before
quarter-time. A skirmish erupted in the second term after Brennan took exception to a tackle by Mattea Breed on Laura McClelland. McClelland was gifted a
50-metre penalty following a bump against Sarah Hosking by Sophie Locke and converted from inside the goal square while players were occupied by the tussle.
</item 13>

<item 14>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 14>

<item 15>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“But I go out with on a happy note. I love these girls and I’ll be over at the pancake pocket [Pancake Parlour at Whitten Oval] watching them from now on and
have a few bevs on the sideline.” Roos’ lock in top-four spot; Dogs have issues to work through in off-season North Melbourne secured third spot on the ladder
and the double chance in finals after downing the Western Bulldogs by a whopping 46 points. They will head into finals with momentum but a glaring issue to work
on: kicking accuracy. Despite the comfortable win, they could have done a lot more damage after booting double the number of behinds to goals.

Star midfielder Ash Riddell, who had 36 touches and seven clearances, said she was excited to go up against Melbourne or Adelaide in the first weekend of finals
depending on other results this round, as they were the only two sides who beat them this year.
</item 15>

<item 16>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Comeback kings Richmond take the points in Cairns A late goal by Emelia Yassir has helped Richmond to their first win in four games, putting their AFLW finals
campaign back on track with an 11-point victory over Hawthorn. Tight tussle: Richmond’s Emelia Yassir is tackled. Credit: Getty Images Separated by one point at
half-time and down by nine points entering the last quarter, the Tigers pulled off a 6.5 (41) to 4.6 (30) victory to keep their finals hopes alive at Cazaly’s
Stadium in Cairns on Friday night.

Hawthorn were limited to just one point in the decisive final quarter. Midfield duo Monique Conti (35 disposals, seven clearances) and Grace Egan (28, 5) were
typically influential for Richmond, while Hawthorn were led by Emily Bates (18, four) and Jasmine Fleming (18, one). “Richmond were just absolute beasts,
especially their midfielders.
</item 16>

<item 17>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Hanks was pivotal for the Demons and starred with 26 disposals, seven tackles and one goal, while her partner in the midfield Olivia Purcell had 28 touches,
five clearances and five tackles. For the Roos, Ash Riddell led the disposal count with 31, while skipper Emma Kearney was crucial in defence. Melbourne’s
Alyssa Bannan is tackled by Lulu Pullar of the Kangaroos. Credit: Getty Hanks said with the state of competition, they just had to take it week by week, but it
was good to get the win. “We knew we were up against the best tonight,” said Hanks. “It’s a nice feeling tonight.”

Lions roar in big battle Meanwhile, on Saturday afternoon, the Brisbane Lions put an end to Adelaide’s unbeaten run with a three-point victory at Brighton Homes
Arena in a top-four thriller, 5.12 (42) to 6.3 (39), in a spectacle worthy of being a final.
</item 17>

<item 18>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
The Lions’ Orla O’Dwyer speeds away against the Crows. Credit: Getty Images Adelaide started fast but the game evened up as the Lions had the better of the
second quarter with three goals to one. The Crows then took back control in the third with three goals to none, which set the stage for a tense final term. The
fourth quarter was a heated arm wrestle, and scores were tied with four minutes left on the clock. The Lions’ Bella Dawes kicked the only goal of the last term,
but it was a series of behinds in the final minutes that won the game for them, including one from Lily Postlethwaite that broke the stalemate.

Adelaide’s Niamh Kelly navigates her way through traffic against brisbane. Credit: Getty The two sides have had intense battles in the competition’s history,
including two grand finals, and this was no different.
</item 18>

<item 19>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
The Dogs have plenty to work on for next season as they look to regain a competitive edge, which was lost after a finals berth last year. Along with being in a
rebuild, they were badly injury-hit this season and used 32 players, more than any other club. Suns secure finals spot AAP Unsung hero Lucy Single has helped
Gold Coast book their second AFLW finals appearance with a 14-point win over Essendon. Losing midfield powerhouse Charlie Rowbottom to an ankle injury proved no
obstacle in the Suns’ maiden clash with the Bombers, as they finished the regular season in fifth spot after a 6.3 (39) to 3.7 (25) victory to kick off pride
round at Great Barrier Reef Arena on Friday.

While their men’s side have yet to make a finals appearance, this is the second time Gold Coast’s women have clinched a play-off spot, having also done so in
the cancelled 2020 season.
</item 19>

<item 20>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
“I can name the whole team. Everyone had their second in the last couple minutes.

“That was such a team effort, everyone had a moment there when they had to put their head over the footy and lay a tackle and not let it slip.” Swanson (25
disposals, eight clearances) stood out for the Eagles, while Bombers workhorse Maddy Prespakis (30 disposals, nine clearances) was typically prominent in the
midfield alongside Georgia Nanscawen (21, six). With two rounds left, West Coast have a chance to build their winning streak when they meet the bottom of the
table Western Bulldogs next round. Slipping one place to sixth, Essendon have must-win matches against Carlton and Gold Coast on the road to the finals. The
Bombers will be without Steph Cain, who was sidelined with a concussion after an elbow to the head from Swanson.
</item 20>
=== END ITEM id=c54645abf4c355d9 ===

=== ITEM id=3d74c7ec144f4390 ===
Gold sentences and facts:
Sentence 0: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 0.0: The party referred to as 'it' did not release the full model, Gemini Ultra.
  Fact 0.1: Only a 'lite' version called Gemini Pro was released.
Sentence 1: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 1.0: The Justice Department is focused on the deals Google makes.
  Fact 1.1: Google makes deals with Apple.
  Fact 1.2: Google makes deals with Samsung.
  Fact 1.3: Google makes deals with Mozilla.
  Fact 1.4: Google makes deals with many others.
  Fact 1.5: The Justice Department is focused on ensuring Google is the default search engine on practically every platform.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 1>

<item 2>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 3>

<item 4>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Most read

AI, faked: Google unveiled a new flagship AI model this week called Gemini. But it didn’t release the full model, Gemini Ultra — only a “lite” version called
Gemini Pro. In a press briefing and blog posts, Google touted Gemini’s coding capabilities and multimodal prowess, claiming that the model can understand
images, audio and videos just as well as text. But Gemini Pro — which is strictly text-in, text-out — has proven to be mistake-prone. And in a worse look for
Google, the company was caught faking a Gemini demo by tuning text prompts with still images off camera.

Offensive notebooks: In another Google PR blunder, people who attended the company’s K&I Black Summit in August were given third-party notebooks containing
highly insensitive language. My colleague Dominic-Madori writes that the inside of the notebooks were printed with the phrase “I was just cotton the moment, but
I came back to take your notes” (emphasis ours).
</item 4>

<item 5>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Now, onto the roundup!

A window into Google’s most popular search queries

Judge Amit Mehta ruled to make a list public that provides a glimpse of which search terms make Google the most money. The list of popular search terms ordered
by revenue includes 20 terms that were lucrative for the week of September 22, 2018. Information like revenue per search term, how many queries each of those
terms got, along with a separate list of popular search terms ordered by queries (not revenue), were all redacted. The list we can see is as follows:

iPhone 8

iPhone 8 plus

Auto insurance

Car insurance

Cheap flights

Car insurance quotes

DirecTV

Online colleges

AT&T

Hulu

iPhone

Uber

Spectrum

Comcast

Xfinity

Insurance quotes

Free credit report

Cheap car insurance

AARP

LifeLock

There is, in reality, little surprise here.
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 6>

<item 7>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 10>

<item 11>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 11>

<item 12>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
A simple question like this with a simple answer like this, and it still got it WRONG. pic.twitter.com/5GowXtscRU — Vitor de Lucca 🏳️‍🌈 /
threads.net/@vitor_dlucca (@vitor_dlucca) December 7, 2023

Note that Gemini Pro claims incorrectly that Brendan Gleeson won Best Actor last year, not Brendan Fraser — the actual winner.

I tried asking the model the same question and, bizarrely, it gave a different wrong answer:

“Navalny,” not “All the Beauty and the Bloodshed,” won Best Documentary Feature last year; “All Quiet on the Western Front” won Best International Film; “Women
Talking” won Best Adapted Screenplay; and “Pinocchio” won Best Animated Feature Film. That’s a lot of mistakes.

Science fiction author Charlie Stross found many more examples of confabulation in a recent blog post.
</item 12>

<item 13>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 13>

<item 14>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Google’s software is constantly crawling the web and updating its central index, which itself is more than 100 million gigabytes in size. When you enter a
query, Search uses a few quick processes to decide the meaning of your question, including what kind of information you might be after. Then, beyond just
looking for sites with the words you typed in, the system uses machine learning to rank results according to what it thinks you’re looking for, prioritising
high quality and easy to use sites. If you allow it, Google does bring in some information it knows about you to further refine results, including your location
and web history. If you’re looking to turn this off, it’s in your Google account settings called “Web & App Activity”. Google says it does not infer personal
information like race or religion to shape results. Loading Google works with independent “search quality raters” to gauge the effectiveness of its processes,
and says it ran almost 900,000 quality tests in 2022.
</item 14>

<item 15>
The restaurant nearest Google (The Verge, 2023-10-26T13:00:00+00:00)
Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence. To improve local
ranking, Google encourages businesses to provide a trove of information on their profile. Owners should have basics like updated hours of operation and accurate
location details, but they should also respond to reviews, add photos, and even post individual items in stock to their Google page.

“I don’t think they’re really great marketing campaigns”

The prominence measure in Maps includes data from the web — links to the business, how the business ranks in traditional search, and press coverage, for
example. In this way, SEO impacts how a business appears in local search results, and news articles and links are one way Google tries to determine the
relevance of a page. (It’s why reporters’ inboxes are overrun with spammy requests to buy “guest posts” on our sites from marketers.)
</item 15>

<item 16>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 16>

<item 17>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 17>

<item 18>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Google spent its 45 minutes attempting to explain that its dominance over the Android app market isn’t anything nefarious but simply the natural outcome of
Google fiercely competing with the iPhone and its iOS App Store, where Google would like the court to believe that competition truly lies.

If Google can convince the jury of that, it could be a winning argument in the case — because obviously, Google doesn’t have a monopoly on app stores or phones
in general. “You cannot separate the quality of a phone from the quality of the apps in its app store, and that means Google and Apple compete against each
other,” began Google lead attorney Glenn Pomerantz.

But Google wound up spending much of its opening statement attempting to explain away its seemingly bad behavior as normal business practices and didn’t always
succeed out of the gate.
</item 18>

<item 19>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 19>

<item 20>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
the list of blue links in search results that are not sponsored. “Ad keyword matching is a long-standing and well-known process that is designed to connect
people to relevant ads. A separate process, which has nothing to do with ads, is used to match organic results to a query,” he said. “It’s no secret that Google
Search looks beyond the specific words in a query to better understand their meaning ... If you make a spelling mistake, or search for a term that’s not on a
page but where the page has a close synonym, or if you aren’t even sure exactly how to search for something, our meaning matching systems help.” It is true that
some search queries result in more ads than others, and what ads you get can vary wildly depending on your request, which is the result of the keyword matching
Google uses to sell ads. But Google denies pushing users towards the most monetisable subjects.
</item 20>

<item 21>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 21>

<item 22>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
(Among other mistruths, Gemini Pro said that Stross contributed to the Linux kernel; he never has.)

Translation doesn’t appear to be Gemini Pro’s strong suit, either. It struggles to give a six-letter word in French:

FYI, Google Gemini is complete trash. pic.twitter.com/EfNzTa5qas — Benjamin Netter (@benjaminnetter) December 6, 2023

When I ran the same prompt through Bard (“Can you give me a 6-letters word in French?”), Gemini Pro responded with a seven-letter word instead of a five-letter
one — which gives some credence to the reports about Gemini’s poor multilingual performance.

What about summarizing news? Surely Gemini Pro, with Google Search and Google News at its disposal, can give a recap of something topical? Not necessarily.

It seems Gemini Pro is loath to comment on potentially controversial news topics, instead telling users to… Google it themselves.

I tried the same prompt and got a very similar response.
</item 22>

<item 23>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 23>

<item 24>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 24>

<item 25>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 25>

<item 26>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 26>

<item 27>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
I tried the same prompt and got a very similar response. ChatGPT, by contrast, gives a bullet-list summary with citations to news articles:

Interestingly, Gemini Pro did provide a summary of updates on the war in Ukraine when I asked it for one. However, the information was over a month out of date:

Google emphasized Gemini’s enhanced coding skills in a briefing earlier this week. Perhaps it’s genuinely improved in some areas — posts on X suggest as much.
But it also appears that Gemini Pro struggles with basic coding functions like this one in Python:

Tried gemini based Bard, and well, it still can't write intersection of two polygons. It's one of those rare relatively simple to express functions that wasn't
ever implemented in python, there is no stack overflow post, and all these models fail on it.
</item 27>

<item 28>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
This week, Google took the wraps off of Gemini, its new flagship generative AI model meant to power a range of products and services including Bard, Google’s
ChatGPT competitor. In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds
the performance of other leading gen AI models like OpenAI’s GPT-4.

But the anecdotal evidence suggests otherwise.

A “lite” version of Gemini, Gemini Pro, began rolling out to Bard yesterday, and it didn’t take long before users began voicing their frustrations with it on X
(formerly Twitter).

The model fails to get basic facts right, like 2023 Oscar winners:

I'm extremely disappointed with Gemini Pro on Bard. It still give very, very bad results to questions that shouldn't be hard anymore with RAG. A simple question
like this with a simple answer like this, and it still got it WRONG.
</item 28>

<item 29>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 29>

<item 30>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 30>

<item 31>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 31>

<item 32>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 32>

<item 33>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 33>

<item 34>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 34>

<item 35>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 35>

<item 36>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The process of this test did remind me how much data Google collects — not just my searches but anything I say to a smart speaker, any search result or ad that
I click — which will be a point in favour of DuckDuckGo for some, even if it didn’t perform as well and is harder to read. The data collection does also mean
that Google (not just search but also mobile widgets and other services) personalises to me more effectively. Loading Bing has overhauled its layout over the
past year to be very graphic and augmented by generative AI explanations from its chatbot. I like the idea of being able to ask follow-up questions in searches
this way, but in practice it’s not really there yet. In a search for “kids BMX bike green” (Google did pretty well and DuckDuckGo ignored the green part), Bing
Chat offered a comparison of a few different options but they were too expensive. I asked, “any under $200?” and it replied with more options that were all over
$300.
</item 36>

<item 37>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 37>

<item 38>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 38>

<item 39>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 39>

<item 40>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 40>

<item 41>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 41>

<item 42>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.
</item 42>

<item 43>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 43>

<item 44>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
Bard on 3 tries(Second Video). pic.twitter.com/cYd9hepcgT — Edison Ade (@buzzedison) December 6, 2023

Just tested Google's Bard with Gemini Pro update. No bugless snake game on 1st try; reported, asked to fix—couldn't. Tried ChatGPT 3.5 free version, got correct
bug-free code on the first attempt! 🚀🐍 #ChatGPT #Bard #Gemini pic.twitter.com/WQfilgG21D — N KIRAN KUMAR (@NKIRANKUMARS1) December 6, 2023

And, as with all generative AI models, Gemini Pro isn’t immune to “jailbreaks” — i.e. prompts that get around the safety filters in place to attempt to prevent
it from discussing controversial topics.
</item 44>

<item 45>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 45>

<item 46>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 46>

<item 47>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 47>

<item 48>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
pic.twitter.com/RKjmkEw2Qr — Filip Piekniewski🌻 🐘:@filippie509@techhub.social (@filippie509) December 6, 2023

And these:

Trying out Gemini Pro: it is pretty disappointing for my example. I asked it to make an analog clock using HTML like this one that ChatGPT made. It can cite
some code from Github but it's off by a few ms… pic.twitter.com/neb42Vzm3m — Mohsen Azimi (@mohsen____) December 7, 2023

GPT 4 still greater than Gemini Pro. Created Tic Tac Toe game with ChatGPT and Bard(Running on Gemini Pro) See video for the result. ChatGPT wrote the code on
first try(First Video). Bard on 3 tries(Second Video).
</item 48>

<item 49>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
He hoped to make Google and Apple’s services so connected as to be inseparable.

The way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention all that juicy data —
and Apple would get more revenue, as a result of the lucrative search agreement the two signed.

For whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.

Google’s deleted chat logs

During Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In February, the
DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually changed the
setting.

Pichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according to CNBC.
</item 49>

<item 50>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 50>
=== END ITEM id=3d74c7ec144f4390 ===
