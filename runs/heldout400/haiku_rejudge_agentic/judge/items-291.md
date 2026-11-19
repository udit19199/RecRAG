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

=== ITEM id=597b53a6e9de923f ===
Gold sentences and facts:
Sentence 0: They want more than three points, but it's early enough that they absolutely could settle for a field goal if it got to that point.
  Fact 0.0: The team in question wants more than three points.
  Fact 0.1: It is still early enough in the game.
  Fact 0.2: The team in question absolutely could settle for a field goal if the game got to that point.
Sentence 1: 11:03 p.m. — The Vikings are using their two timeouts, but there isn't much they'll be able to do once this hits fourth down.
  Fact 1.0: At 11:03 p.m., the Vikings are using their two timeouts.
  Fact 1.1: There isn't much the Vikings will be able to do once the situation reaches fourth down.
Sentence 2: The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running game for Kevin O'Connell and the defense overall under Brian Flores.
  Fact 2.0: Josh Dobbs is a new player on the Vikings.
  Fact 2.1: Josh Dobbs keeps producing the same offensive results the Vikings enjoyed with Kirk Cousins.
  Fact 2.2: Josh Dobbs's results have come with more help from the running game under Kevin O'Connell.
  Fact 2.3: Josh Dobbs's results have come with more help from the overall defense under Brian Flores.

Retrieved items:
<item 1>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
9:21 p.m.: Joshua Dobbs and center Garrett Bradbury have had a few bad exchanges when the newly acquired QB is under center. Dobbs just fumbled his second snap
of the game but has recovered both of them.

Vikings 10, Broncos 6

9:18 p.m.: FIELD GOAL — Will Lutz gets another 31-yarder to go for the Broncos. Russell Wilson found Jerry Jeudy for a five-yard gain on 3rd & 7, coming just
short of the line to gain. Head coach Sean Payton elected to kick the field goal instead of going for it on Minnesota's 12-yard line.

9:14 p.m.: Russell Wilson and Courtland Sutton connect on a 33-yard pitch and catch to give the Broncos their first big play of the game! Denver is in the red
zone for the first time tonight.

Vikings 10, Broncos 3

9:08 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Garrett Bradbury, Vikings, Russell Wilson, Joshua Dobbs, Will Lutz, Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night
Football' game, Sporting News, Jerry Jeudy, Sean Payton, NFL 'Sunday Night Football' game, Broncos
</item 1>

<item 2>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
The Vikings decided to go for it and Joshua Dobbs picked up the first down on a QB sneak.

10:54 p.m.: Ty Chandler continues to rumble. The spell RB picked up a massive gain on 3rd & 20 and it appeared that he got the first down, but the spot on the
field is under review. This could be a crucial call.

10:48 p.m.: It looked like it was going to be a three-and-out for the Vikings offense but a fake punt has kept the drive alive! Ty Chandler picked up 31 yards
on a gutsy call from head coach Kevin O'Connell to bring the ball into Denver's territory.

Vikings 17, Broncos 15

10:43 p.m.: FIELD GOAL — Will Lutz knocks one in from 28 yards, his fifth made field goal of the day. Lutz has been the Broncos' lone source of offense,
bringing Denver within one score with 10:37 to play in the game.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Ty Chandler, Vikings, Joshua Dobbs, Vikings vs. Broncos NFL 'Sunday Night Football' game, Will Lutz, Vikings vs. Broncos live score, updates,
highlights from NFL 'Sunday Night Football' game, Kevin O'Connell, Sporting News, Broncos
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Minnesota, Josh Dobbs, Bears vs. Vikings game, Adrian Peterson, Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night
Football' game, U.S. Bank Stadium, Sporting News, Arizona, 2024 NFL Draft, Bears, T.J. Hockenson
</item 3>

<item 4>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
8:44 p.m.: The Vikings' defense responds with a stop of their own. A quick three-and-out for Russell Wilson and the Broncos. Minnesota will take over from its
own 35 after a 12-yard punt return from Brandon Powell. Broncos lead 3-0 with 5:01 to play in the first quarter.

8:38 p.m.: Joshua Dobbs had the Vikings' offense moving but the Broncos' defense came up with a big stop on 3rd & 8. Minnesota had picked up 28 yards in six
plays but Dobbs was nearly picked off on third down, forcing a punt. Denver will take over from its own 25-yard line after a touchback.

8:32 p.m.: Joshua Dobbs was evaluated in the medical tent after taking a big hit on the last drive, but he did not miss a snap at QB for the Vikings.

Broncos 3, Vikings 0

8:30 p.m.

Graph facts:
- Vikings -MENTIONS-> Minn.
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> Vikings

Entities: Vikings, Russell Wilson, Brandon Powell, Joshua Dobbs, Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game, NFL,
Sporting News, NFL 'Sunday Night Football' game, Broncos
</item 4>

<item 5>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Raiders
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Raiders, Vikings, Josh Dobbs, Dallas, fantasy football, Ravens, Sporting News, Geno Smith, Bears, 49ers, Matthew Stafford, Nick Mullens, Seattle,
Seattle's one-possession loss to Dallas, Pass defenses
</item 5>

<item 6>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
He's responsible for all of Denver's points as it trails 10-9.

9:35 p.m.: Russell Wilson has the Broncos' offense moving. He's delivered four complete passes for 46 yards, bringing Denver down to Minnesota's 34-yard line
with 11 seconds on the clock. The Broncos have one timeout remaining.

9:27 p.m.: Alexander Mattison was stuffed on a three-yard run, then Joshua Dobbs couldn't find his RB on a swing route on 3rd down, forcing a punt. The Broncos
will start their drive from their own 25 with 1:11 remaining in the half after a touchback on the punt.

9:24 p.m.: Joshua Dobbs connected on a 17-yard pass to Jordan Addison and a 10-yard pass to TJ Hockenson to bring Minnesota to its own 48 at the two-minute
warning. It will be 2nd & 5 coming out of the break.

9:21 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings vs. Broncos game, TJ Hockenson, Vikings, Russell Wilson, Joshua Dobbs, Vikings vs. Broncos live score, updates, highlights from NFL
'Sunday Night Football' game, Alexander Mattison, Jordan Addison, Sunday Night Football, Sporting News, Broncos
</item 6>

<item 7>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Joshua Dobbs, Chicago, Minnesota, Fields, Bears vs. Vikings 'Monday Night Football' game, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, Bears
</item 7>

<item 8>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
If you tuned into "Sunday Night Football" expecting to see some offense, you tuned in on the wrong weekend.

The Jets and Raiders were locked into an old-school, defense-heavy primetime clash that featured nothing but six field goals through the first three quarters of
the game. It wasn't until 50 seconds into the fourth quarter that Las Vegas finally picked up the first touchdown, with Michael Mayer making a leaping grab in
the end zone to put the Raiders up 16-9.

In such a low-scoring game, that touchdown felt, even as early as it was, that it could be a game-winner. And indeed, it was as the Raiders held the Jets to
just a field goal the rest of the game and picked off Zach Wilson late to stymy New York in a 16-12 win.

MORE: What to know about Raiders 'SNF' hero Robert Spillane

It wasn't exactly a quarterback showdown that many might have expected in the preseason when this game appeared to be Aaron Rodgers vs.

Graph facts:
- Michael Mayer -PARTICIPATED_IN-> Jets vs. Raiders game
- Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end -MENTIONS-> Michael Mayer
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Michael Mayer
- Michael Mayer -PARTICIPATED_IN-> Monday Night Football game
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Michael Mayer, Raiders, NFL, Zach Wilson, Jets, Sporting News, Aaron Rodgers, Sunday Night Football, Robert Spillane, Jets vs. Raiders live score,
updates, highlights from NFL 'Sunday Night Football' game
</item 8>

<item 9>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Vikings 10, Broncos 3

9:08 p.m.: FIELD GOAL — Greg Joseph drills a 34-yarder to give the Vikings a 10-3 lead with 8:09 to play in the half. The Vikings' offense stalled out in the
red zone as Joshua Dobbs was sacked for an 11-yard loss on 3rd & Goal.

9:06 p.m.: Joshua Dobbs has some magic in him. In a cloud of smoke in the pocket, Dobbs found TE TJ Hockenson on a 29-yard gain to pick up a first down on 3rd &
2. The big play brings the Vikings into the red zone.

9:00 p.m.: Another three-and-out for Russell Wilson and the Broncos. On 3rd & 3, Wilson was under heavy pressure and threw the ball away as he was nearly
brought down for a sack.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Greg Joseph, TJ Hockenson, Vikings, Russell Wilson, Joshua Dobbs, Vikings vs. Broncos NFL 'Sunday Night Football' game, Vikings vs. Broncos live
score, updates, highlights from NFL 'Sunday Night Football' game, Sporting News, Broncos
</item 9>

<item 10>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Denver punted to Minnesota's 25, where Joshua Dobbs and the offense will take over.

Vikings 7, Broncos 3

8:56 p.m.: TOUCHDOWN! Joshua Dobbs, are you kidding me?! Dobbs scrambled out right and looked like he had nowhere to go. The QB shed a tackle on the sidelines
and regained his composure, finding TE Josh Oliver wide open in the end zone for a touchdown after the play broke down.

End of first quarter: Broncos 3, Vikings 0

8:53 p.m.: The Vikings' run game is marching the offense down the field. Spell RB Ty Chandler has collected 29 total yards on this drive while starting RB
Alexander Mattison has picked up 27 yards. Minnesota is down to Denver's 4-yard line as the first quarter comes to a close.

8:44 p.m.: The Vikings' defense responds with a stop of their own.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings vs. Broncos game, Josh Oliver, Ty Chandler, Vikings, Joshua Dobbs, Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday
Night Football' game, Alexander Mattison, Sunday Night Football, Sporting News, Broncos
</item 10>
=== END ITEM id=597b53a6e9de923f ===

=== ITEM id=21f2e1e70ca85cc9 ===
Gold sentences and facts:
Sentence 0: But it was ChatGPT’s explosion into public consciousness that thrust Altman into the spotlight as a face of generative AI — technology that can produce novel imagery, passages of text and other media.
  Fact 0.0: ChatGPT's explosion into public consciousness thrust Altman into the spotlight as a face of generative AI.
  Fact 0.1: Generative AI is technology that can produce novel imagery, passages of text and other media.
Sentence 1: “There was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic,” said an Anthropic spokesperson, Sally Aldous.
  Fact 1.0: An Anthropic spokesperson, Sally Aldous, said there was no attempt to remove Sam Altman from OpenAI by the co-founders of Anthropic.
Sentence 2: More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
  Fact 2.0: Sam is one of the most brilliant and impactful people the industry has ever had.
  Fact 2.1: Sam is one of the most generous and caring people the speaker knows.
  Fact 2.2: Being one of the most generous and caring people is more important than being one of the most brilliant and impactful people.
Sentence 3: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 3.0: The prevailing theory is based on the board's language.
  Fact 3.1: The prevailing theory is based on the way giant tech companies work.
  Fact 3.2: The prevailing theory is circulating right now.

Retrieved items:
<item 1>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT: Everything you need to know about the AI-powered chatbot

ChatGPT, OpenAI’s text-generating AI chatbot, has taken the world by storm. What started as a tool to hyper-charge productivity through writing essays and code
with short text prompts has evolved into a behemoth used by more than 92% of Fortune 500 companies for more wide-ranging needs. And that growth has propelled
OpenAI itself into becoming one of the most-hyped companies in recent memory, even if CEO and co-founder Sam Altman’s firing and swift return raised concerns
about its direction and opened the door for competitors.

What does that mean for OpenAI, ChatGPT and its other ambitions? The fallout is still settling, but it might empower competitors like Meta and its LLaMA family
of large language models, or help other AI startups get attention and funding as the industry watches OpenAI implode and put itself back together.

Graph facts:
- Founders, are events useful? -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT

Entities: LLaMA family, Fortune 500 companies, ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Meta, TechCrunch, OpenAI, AI
startups, Sam Altman
</item 1>

<item 2>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI CEO Sam Altman had previously pointed to industry-wide GPU shortages for the service’s spotty performance. Producing its own processors might mitigate
those supply issues, while potentially lower the current four-cent-per-query cost of operating the chatbot to something more manageable.

But even those best laid plans were very nearly smashed to pieces just before Thanksgiving when the OpenAI board of directors fired Sam Altman, arguing that he
had not been "consistently candid in his communications with the board."

That firing didn't take. Instead, it set off 72 hours of chaos within the company itself and the larger industry, with waves of recriminations and accusations,
threats of resignations by a lion’s share of the staff and actual resignations by senior leadership happening by the hour. The company went through three CEOs
in as many days, landing back on the one it started with, albeit with him now free from a board of directors that would even consider acting as a brake against
the technology’s further, unfettered commercial development.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, How OpenAI's ChatGPT has changed the world in just a year, OpenAI board of directors, GPU shortages, Sam Altman's firing by the OpenAI board,
OpenAI, Sam Altman, Engadget
</item 2>

<item 3>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman posted Friday on X: “i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved
working with such talented people. will have more to say about what’s next later.”

The Associated Press and OpenAI have a licensing and technology agreement allowing OpenAI access to part of the AP’s text archives.

Altman helped start OpenAI as a nonprofit research laboratory in 2015. But it was ChatGPT’s explosion into public consciousness that thrust Altman into the
spotlight as a face of generative AI — technology that can produce novel imagery, passages of text and other media. On a world tour this year, he was mobbed by
a crowd of adoring fans at an event in London.

He’s sat with multiple heads of state to discuss AI’s potential and perils.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, X, AI, Fortune, OpenAI, The Associated Press, event in London, firing Altman, going rogue, OpenAI board, OpenAI's ex-chairman accuses board
of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did', London, Sam Altman, generative AI
</item 3>

<item 4>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
People flat out lost their minds over the new AI and its CEO, Sam Altman. Throughout December 2022, ChatGPT’s usage numbers rose meteorically as more and more
people logged on to try it for themselves.

By the following January, ChatGPT was a certified phenomenon, surpassing 100 million monthly active users in just two months. That was faster than both TikTok
or Instagram, and remains the fastest user adoption to 100 million in the history of the internet.

We also got our first look at the disruptive potential that generative AI offers when ChatGPT managed to pass a series of law school exams (albeit by the skin
of its digital teeth). Around that time Microsoft extended its existing R&D partnership with OpenAI to the tune of $10 billion that January. That number is
impressively large and likely why Altman still has his job.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, Microsoft, a series of law school exams, extension of existing R&D partnership, How OpenAI's ChatGPT has changed the world in just a year,
OpenAI, Instagram, TikTok, Sam Altman, generative AI, Engadget
</item 4>

<item 5>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
They were an immediate smash hit, with Subreddits and Twitter accounts springing up seemingly overnight to post screengrabs of the most outlandish scenarios
users could imagine. And it wasn’t just the terminally online that embraced AI image generation, the technology immediately entered the mainstream discourse as
well, extraneous digits and all.

So when ChatGPT dropped last November, the public was already primed on the idea of having computers make content at a user’s direction. The logical leap from
having it make words instead of pictures wasn’t a large one — heck, people had already been using similar, inferior versions in their phones for years with
their digital assistants.

Q1: [Hyping intensifies]

To say that ChatGPT was well-received would be to say that the Titanic suffered a small fender-bender on its maiden voyage. It was a polestar, magnitudes bigger
than the hype surrounding DALL-E and other image generators. People flat out lost their minds over the new AI and its CEO, Sam Altman.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> DALL-E
- Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs -MENTIONS-> DALL-E
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT

Entities: DALL-E, ChatGPT, AI image generation, Titanic, How OpenAI's ChatGPT has changed the world in just a year, OpenAI, Sam Altman, Twitter, Engadget
</item 5>

<item 6>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.

Graph facts:
- One year later, ChatGPT is still alive and kicking -MENTIONS-> The New York Times
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Amazon Q
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Stella Biderman
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Booz Allen Hamilton

Entities: One year later, ChatGPT is still alive and kicking, security assessments, Steve Wozniak, open letter calling for the immediate pause of large-scale AI
experiments, large-scale AI experiments, OpenAI's models, AI-powered chatbots, cookbook collection, company's proprietary codebases, developers, ChatGPT abuse,
Signing an open letter calling for the immediate pause of large-scale AI experiments, ChatGPT, Apple, GPTs
</item 6>

<item 7>
Graph query row:
article: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
published_on: 2023-10-07T20:15:26+00:00
publisher: TechCrunch
passages: ['Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nHiya, folks,
and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the past few
days, and — we humbly submit — it’s a darn useful resource for folks on the go.\n\nThis week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware
event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok
testing an ad-free subscription plan, and LinkedIn going big on AI tools. And that’s not all.\n\nIf you haven’t, sign up here to get WiR in your inbox every
Saturday. And if you have, our thanks. Now, let’s get on with the news.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an
ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nAnd if you have, our thanks. Now, let’s get on with the news.\n\nMost read\n\nAltman backs teen
entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating workflows in
“previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and converts the
instructions to pseudo-code in real time.\n\nGoogle unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant
showed off what it’s been working on. Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold,
Android 14, Pixel Buds, Google Assistant, Bard, Pixel Watch 2 and other goodies.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests
an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nHands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked
what he saw. While very similar to last year’s model (the Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features
made it enough of an upgrade to (potentially) warrant a purchase — minus the underutilized temperature sensor. Stay tuned for his full review.\n\nTurmoil at
Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board, calling
recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a report from
CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.', 'Sam Altman backs
teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nFor $4.99, subscribers get an ad-free
experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that it’s piloting the plan in a single,
English-speaking market outside the U.S. for now.\n\nLinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job
hunting, marketing and sales products, Ingrid writes. They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it
throughout; an AI-powered LinkedIn Learning coach; and a new AI-powered tool for marketing campaigns.\n\nMusk comes clean about X’s metrics — maybe: In
September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million posts every day, excluding retweets. But speaking
at an event this week, X CEO Linda Yaccarino offered a contradictory figure.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an
ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nShe claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.\n\nFormer
NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its remaining
staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than $400 million
in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of an initial
spike.\n\nAudio\n\nOn the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further
than TechCrunch’s roster, which covers the world of startups, the blockchain and more.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok
tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nOn Equity this week, the crew talked about the SBF trial; deals from VR firms Rainforest, At One
Ventures, Section 32 and Greylock, where venture funding has declined; and how Fearless Fund, a firm founded to invest in women of color, is being barred from
awarding grants to Black women founders.\n\nMeanwhile, Found featured Esther Rodriguez-Villegas from Acurable, a medical device company that makes
patient-friendly wearable devices to diagnose and manage respiratory conditions at home. As a career-long academic, Rodriguez-Villegas talks about how she never
intended to be a founder until she learned about how the currently available medical devices make it extremely difficult to detect and treat diseases like sleep
apnea and epilepsy.\n\nAnd over on Chain Reaction, Jacquelyn did a crossover episode with Alex about the SBF trial.', 'Sam Altman backs teens’ startup, Google
unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nJacquelyn has been on the ground at the Southern District of New
York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots to talk about.\n\nTechCrunch+\n\nTC+ subscribers get access to
in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up. Here are a few highlights from
this week:\n\nInside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone
who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to
commit fraud or steal and basically got in over his head.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
(TechCrunch, 2023-10-07T20:15:26+00:00)\nBattery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software
that can be slipped into just about any lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.\n\nWhat lies
beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched on where
startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.']
mentioned_entities: ['SBF Trial', 'Google’s hardware event', 'Pixel 8 Pro', 'Flexport drama', 'FTX fallout', 'Gmail', 'ad-free subscription plan', 'Induced AI',
'Manish', 'Google’s annual hardware event', 'Pixel Fold', 'Pixel Buds', 'Pixel Watch 2', 'teens’ startup', 'ad-free tier', 'Pixel 7 Pro', 'recent reporting on
Flexport', 'Ingrid', 'Recruiter', 'LinkedIn Learning coach', 'AI-powered tool for marketing campaigns', 'TikTok’s ad-free plan', 'X users’ daily posts and
content generation', 'X users’ daily posts and content generation figures', 'Pixel 8', 'IronNet', 'Keith Alexander', 'NSA', 'going public', 'blockchain',
'Equity', 'At One Ventures', 'Section 32', 'Greylock', 'Fearless Fund', 'grants', 'Black women founders', 'Found', 'Esther Rodriguez-Villegas', 'Acurable',
'patient-friendly wearable devices', 'respiratory conditions', 'epilepsy', 'Chain Reaction', 'Jacquelyn', 'Southern District of New York courthouse',
'Bankman-Fried', 'SBF and FTX trial', 'Breathe Battery Technologies', 'software', 'lithium-ion battery', 'faster charging speeds', 'greater longevity', 'Anna',
'10 investors', 'survey of 10 investors', 'oligopoly dynamics', 'first-mover advantages', 'proprietary data', 'Linda Yaccarino', 'Darrell', 'ChatGPT', 'AI
tools', 'LinkedIn', 'Rainforest', 'Virginia', 'TechCrunch+', 'Tim', 'the defense', 'Bard', 'X', 'Christine', 'AI', 'spam', 'Elon Musk', 'startups', 'FTX',
'Google', 'Rebecca', 'the prosecution', 'TikTok', 'cybersecurity', 'CNBC', 'venture funding', 'sleep apnea', 'Alex', 'Sam Altman', 'Google Assistant', 'Android
14', 'U.S.', 'fraud', 'Dave Clark', 'Flexport', 'Amazon']
</item 7>

<item 8>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
Countless other ChatGPT rivals and derivatives have arrived to market since, most recently Amazon Q, a more business-oriented take on ChatGPT. DeepMind,
Google’s premier AI research lab, is expected to debut a next-gen chatbot, Gemini, before the end of the year.

Stella Biderman, an AI researcher at Booz Allen Hamilton and the open research group EleutherAI, told me that she doesn’t see ChatGPT as an AI breakthrough per
se. (OpenAI, which has released dozens of research papers on its models, tellingly never released one on ChatGPT.) But, she says, ChatGPT was a bonafide “user
experience breakthrough” — taking generative AI mainstream.

“The primary impact [ChatGPT] has had [is] encouraging people training AIs to try to mimic it, or encouraging people studying AIs to use it as their central
object of study,” Biderman said.

Graph facts:
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Data.ai
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> The New York Times
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT

Entities: Amazon Q, Stella Biderman, Booz Allen Hamilton, EleutherAI, AI breakthrough, user experience breakthrough, One year later, ChatGPT is still alive and
kicking, ChatGPT, DeepMind, TechCrunch, Google, OpenAI, Gemini, generative AI
</item 8>

<item 9>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q3: [Pushback intensifies]

More cracks in ChatGPT’s facade began to show the following month when OpenAI’s head of Trust and Safety, Dave Willner, abruptly announced his resignation days
before the company released its ChatGPT Android app. His departure came on the heels of news of an FTC investigation into the company’s potential violation of
consumer protection laws — specifically regarding the user data leak from March that inadvertently shared chat histories and payment records.

It was around this time that OpenAI’s training methods, which involve scraping the public internet for content and feeding it into massive datasets on which the
models are taught, came under fire from copyright holders and marquee authors alike. Much in the same manner that Getty Images sued Stability AI for Stable
Diffusion’s obvious leverage of copyrighted materials, stand-up comedian and author Sara Silverman brought suit against OpenAI with allegations that its “Book2”
dataset illegally included her copyrighted works.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, How OpenAI's ChatGPT has changed the world in just a year, Dave Willner, ChatGPT Android app, FTC investigation into OpenAI, consumer
protection laws, user data leak, Getty Images, Sara Silverman, Book2 dataset, Getty Images lawsuit against Stability AI, Sara Silverman's lawsuit against
OpenAI, copyright, FTC, Stable Diffusion
</item 9>

<item 10>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In its privacy policy, the ChatGPT maker makes a passing acknowledgement of the objection requirements attached to relying on “legitimate interest” (LI),
pointing users towards more information about requesting an opt out — when it writes: “See here for instructions on how you can opt out of our use of your
information to train our models.”

What controversies have surrounded ChatGPT?

Recently, Discord announced that it had integrated OpenAI’s technology into its bot named Clyde where two users tricked Clyde into providing them with
instructions for making the illegal drug methamphetamine (meth) and the incendiary mixture napalm.

An Australian mayor has publicly announced he may sue OpenAI for defamation due to ChatGPT’s false claims that he had served time in prison for bribery. This
would be the first defamation lawsuit against the text-generating service.

CNET found itself in the midst of controversy after Futurism reported the publication was publishing articles under a mysterious byline completely generated by
AI.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Clyde, methamphetamine, napalm, Australian mayor, CNET, Futurism, bribery,
TechCrunch, defamation, AI, OpenAI, Discord
</item 10>
=== END ITEM id=21f2e1e70ca85cc9 ===

=== ITEM id=1720c490c111ae8b ===
Gold sentences and facts:
Sentence 0: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 0.0: An earlier roundup reported that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 0.1: An earlier roundup reported that Google tried to have Chrome preinstalled on iPhones.
Sentence 1: The company did accomplish that goal, but in doing so, it inadvertently and profoundly changed how the internet looked.
  Fact 1.0: Google accomplished that goal.
  Fact 1.1: In accomplishing that goal, Google inadvertently changed how the internet looked.
  Fact 1.2: In accomplishing that goal, Google profoundly changed how the internet looked.
Sentence 2: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 2.0: The full model, Gemini Ultra, was not released.
  Fact 2.1: Only a 'lite' version called Gemini Pro was released.
Sentence 3: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 3.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 3.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 3.2: The case argues that Google siphons off news publishers' readers.
  Fact 3.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 3.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Yahoo, Baker, search technology,
Mozilla, Firefox, TechCrunch, Google, Bloomberg
</item 1>

<item 2>
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

Graph facts:
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn -MENTIONS-> Publishers
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet

Entities: Knowledge Graph, Publishers, Knowledge Panel, Featured Snippets, class action antitrust suit, misappropriation of Publishers’ content, The Wall Street
Journal, report on Thursday, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, antitrust, TechCrunch, AI,
Google
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Publishers -PARTICIPATED_IN-> class action antitrust suit
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal

Entities: class action antitrust suit, The Wall Street Journal, deal to license news for AI model training, News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line, antitrust, The Atlantic, Axel Springer, TechCrunch, AI, Similarweb, Google, OpenAI
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Helena World Chronicle, U.S. District Court in D.C., Search Generative Experience, Bard AI chatbot, Knowledge Graph, anticompetitive
behavior, U.S. antitrust law, Sherman Act, Arkansas, question-and-answer technologies, News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line, class action lawsuit, TechCrunch, AI technologies
</item 4>

<item 5>
Graph query row:
organization: Alphabet
report: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
related_entity: Helena World Chronicle
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.
- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
-MENTIONS-> Washington Post
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Search Generative Experience

Entities: SGE, Google Bard AI, GoogleBot, Google general search service, AdSense, News Media Alliance, Washington Post, Epic Games lawsuit against Google over
app store antitrust issues, app store antitrust issues, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line,
class action antitrust suit against Google, Epic Games, Allen Institute for AI, TechCrunch, Google
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 8>

<item 9>
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

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- Alphabet -ACCUSED_OF-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.

Entities: Sherman Act, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, U.S. Justice Department,
Hausfeld, Helena World Chronicle, LLC, class action antitrust suit against Google, U.S. Justice Department lawsuit against Google, 2020 Justice Department civil
antitrust suit over search and search advertising, digital ad technologies, Clayton Act, search and search advertising, digital news and reference publishing,
class-wide monetary and injunctive relief, free marketplace of ideas, free press
</item 9>

<item 10>
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

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Mozilla, Apple, TechCrunch, Prabhakar Raghavan, Pichai, trial hearing, court testimony, Google antitrust case, decision made with the consumer
in mind, 5 things we learned so far about the Google antitrust case, Samsung, Google, New York Times, iPhones
</item 10>
=== END ITEM id=1720c490c111ae8b ===
