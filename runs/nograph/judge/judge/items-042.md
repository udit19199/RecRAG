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

This file has 6 items.

=== ITEM id=7a1c5606a544a1cc ===
Gold sentences and facts:
Sentence 0: Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.
  Fact 0.0: Altman's exit is indeed shocking
  Fact 0.1: Altman has been the face of generative AI technology
  Fact 0.2: Arun Chandrasekaran, a Gartner analyst, said this
Sentence 1: Altman has been telling investors that he’s planning to launch a new venture, according to The Information.
  Fact 1.0: Altman has been telling investors that he is planning to launch a new venture
  Fact 1.1: The Information is the source of this report
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory is based on the board's language
  Fact 2.1: The prevailing theory is based on the way giant tech companies work
  Fact 2.2: The prevailing theory is floating around right now

Retrieved items:
<item 1>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
We are collaborating to figure out the details. Thank you so much for your patience through this. — OpenAI (@OpenAI) November 22, 2023

Altman also posted about the new deal, giving some insight into the roller coaster that we’ve all been riding since his firing was revealed Friday. In his
words, his decision to join Microsoft on Sunday “was the best path for me and the team.” Since then, the new board’s composition and Microsoft’s support appear
to have been enough to bring him back to the AI company he co-founded.

Altman and board in talks

OpenAI’s board of directors is reportedly in talks with Sam Altman, ex-Y Combinator president and an OpenAI co-founder, to return to OpenAI as CEO as soon as
this week.
</item 1>

<item 2>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 2>

<item 3>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
The company said its board consists of OpenAI’s chief scientist, Ilya Sutskever, and three non-employees: Quora CEO Adam D’Angelo, tech entrepreneur Tasha
McCauley, and Helen Toner of the Georgetown Center for Security and Emerging Technology.

OpenAI’s key business partner, Microsoft, which has invested billions of dollars into the startup and helped provide the computing power to run its AI systems,
said that the transition won’t affect its relationship.

“We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of AI to our customers,” said an
emailed Microsoft statement.

While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s. He was recruited in 2014 to take lead of
the startup incubator YCombinator.
</item 3>

<item 4>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Later Monday, that number climbed to over 650.

Altman and Brockman considering return

As reported by The Verge, Altman’s move to Microsoft isn’t a done deal — and both Altman and Brockman are still open to returning to OpenAI. That is, if the
remaining board members who initially fired him step aside.

OpenAI board considers merger

OpenAI’s board of directors approached Dario Amodei, the co-founder and CEO of rival large-language model developer Anthropic, about a potential merger of the
two companies, The Information reports. The approach was part of an effort by OpenAI to persuade Amodei to replace Altman as CEO — but Amodei quickly turned
down the CEO offer.

—

November 19

Altman to meet at OpenAI HQ

According to The Information, Altman is expected to meet at OpenAI’s San Francisco headquarters as executives at OpenAI push to have him reinstated as CEO.
</item 4>

<item 5>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
But later on X, formerly Twitter, Brockman posted a message he sent to OpenAI employees in which he wrote, “based on today’s news, i quit.”

In another X post on Friday night, Brockman said Altman was asked to join a video meeting at noon Friday with the company’s board members, minus Brockman,
during which OpenAI co-founder and Chief Scientist Ilya Sutskever informed Altman he was being fired.

“Sam and I are shocked and saddened by what the board did today,” Brockman wrote, adding that he was informed of his removal from the board in a separate call
with Sutskever a short time later.

OpenAI declined to answer questions on what Altman’s alleged lack of candor was about. The statement said his behavior was hindering the board’s ability to
exercise its responsibilities.

Altman posted Friday on X: “i loved my time at openai.
</item 5>

<item 6>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 6>

<item 7>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Sam Altman has been fired from OpenAI, Inc., the 501(c)(3) nonprofit that acts as the governing body for OpenAI, the AI startup behind ChatGPT, DALL-E 3, GPT-4
and other highly capable generative AI systems. He’ll both leave the company’s board of directors and step down as CEO.

In a post on OpenAI’s official blog, the company writes that Altman’s departure follows a “deliberative review process by the board” that concluded that Altman
“wasn’t consistently candid in his communications” with other board members, “hindering its ability to exercise its responsibilities.”

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

“The board no longer has confidence in [Altman’s] ability to continue leading OpenAI,” the blog post reads.
</item 7>

<item 8>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Microsoft has poured $13 billion into OpenAI so far and has what amounts to a 49% stake in the company. The former’s stock price fell more than 1% in the last
30 minutes of trading, after Altman’s departure was announced; Axios reports Microsoft was notified Altman would be leaving “minutes” before the public
announcement.

OpenAI was founded as a nonprofit in 2015, but restructured in 2019 as a “capped-profit” company to raise capital — an acknowledgement of the huge costs
associated with training cutting-edge AI systems. Underlining the point, Altman said in an interview this week with the Financial Times that he “hoped”
Microsoft would increase its investment to help pay for “huge” imminent model training costs.

When contacted via email, OpenAI’s press relations declined to comment beyond the official blog post.
</item 8>

<item 9>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 9>

<item 10>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
one takeaway: go tell your friends how great you think they are. — Sam Altman (@sama) November 18, 2023

Investors pushing for Altman’s return

Investors — furious at the turn of events — are reportedly exerting pressure on OpenAI’s board to reinstate Altman, going so far as to recruit Microsoft.
Nadella is said to be sympathetic.

Board agrees to reverse course — in principle

The Verge reports that the board agreed in principle to resign and to allow Altman and Brockman to return. It waffled, however, missing a deadline yesterday by
which many OpenAI staffers were set to leave the company. Altman is said to be ambivalent about coming back and asking for “significant” governance changes.

—

November 17

Brockman demoted

Brockman says he got a text from Sutskever shortly after noon on Friday asking for a quick call.
</item 10>
=== END ITEM id=7a1c5606a544a1cc ===

=== ITEM id=194f4153bcaadbe1 ===
Gold sentences and facts:
Sentence 0: Likely with the sole intent of keeping Messi fit in that time off, and the added bonus of raking in significant cash, Inter Miami scheduled a two-match tour of China in early November after their regular season concluded.
  Fact 0.0: Inter Miami scheduled a two-match tour of China in early November.
  Fact 0.1: The China tour was scheduled after Inter Miami's regular season concluded.
  Fact 0.2: The likely sole intent of the China tour was keeping Messi fit during that time off.
  Fact 0.3: The China tour had the added bonus of raking in significant cash.
Sentence 1: Eventually, Alvaro Barreal’s 78th-minute goal ended any hope that Inter Miami had to complete a stunning playoff charge, and the defeat coupled with CF Montreal’s big 4-1 victory over Portland means Inter Miami are officially out of the postseason running.
  Fact 1.0: Alvaro Barreal scored a goal in the 78th minute.
  Fact 1.1: Barreal's 78th-minute goal ended any hope Inter Miami had of completing a stunning playoff charge.
  Fact 1.2: Inter Miami's defeat, together with CF Montreal's 4-1 victory over Portland, means Inter Miami are officially out of the postseason running.
Sentence 2: Lionel Messi's arrival at Inter Miami has brought with it a craze for fans to get a glimpse of one of the sport's most talented and legendary players of all time.
  Fact 2.0: Lionel Messi's arrival at Inter Miami has brought a craze for fans to get a glimpse of him.
  Fact 2.1: Lionel Messi is one of the sport's most talented players of all time.
  Fact 2.2: Lionel Messi is one of the sport's most legendary players of all time.

Retrieved items:
<item 1>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Inter Miami season tickets: Prices for 2024

The pricing structure for 2024 represents an enormous increase over the 2023 costs. The cheapest season ticket in 2023 was $485, but that will jump all the way
to $884, which is an increase of 78.8%. There is a discount for fans who renew by October 7.

Meanwhile, a season ticket in the lower-level midfield club area, which includes upscale food and beverage options, cost $3,600 in 2023 but will now run $7,650
for the 2024 season, an increase of 113%, more than doubling year-over-year.

Inter Miami 2024 season ticket prices

Prices below courtesy of Inter Miami season ticketing release via the Miami Herald on September 28, 2023. Sections 2 through B are considered "premium" seating,
while C through L are considered "standard."
</item 1>

<item 2>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
HALFTIME: Inter Miami 0-0 FC Cincinnati

Needing victory in this match to ensure they remain in the MLS playoff picture, Inter Miami completely dominated the confirmed 2023 regular season champs but
somehow did not manage to capitalize on their overwhelming pressure. Inter Miami were so unlucky not to cash in on multiple occasions, hitting the woodwork an
insane three times, all in the first 30 minutes. Lionel Messi may yet have a job to do tonight, currently available off the bench.

FC Cincinnati had one really good look that they missed as well but otherwise were on their heels the entire 45 minutes. Head coach Pat Noonan tells the
broadcast he feels FC Cincinnati had "a lack of courage on the ball" that half, "as bad as its been all year."

Elsewhere, Inter Miami will be nervous about other results from around the league.
</item 2>

<item 3>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Messi joined Inter Miami with the club bottom of the Eastern Conference, and the gap was just too big to overcome. Messi's injury certainly played a role, but
even so, there was just not enough room for error.

It's official: Inter Miami has been mathematically eliminated from MLS playoff contention. pic.twitter.com/u39ZgeXI9M — Tom Bogert (@tombogert) October 8, 2023

Inter Miami vs FC Cincinnati: Second Half

90+3 min: With five minutes of added time on the board, Lionel Messi draws another foul in a very dangerous position, this one much more obvious for the
official to make. It's just outside the penalty area from a central position, and Messi stands over it yet again.

His effort on goal is close to finding the top-left corner, but it's just high, clipping off the very top of the crossbar.
</item 3>

<item 4>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Inter Miami were in simply too big of a hole in the MLS playoff race, one which even Lionel Messi couldn’t drag them out of.

With two matches remaining in the season, Inter Miami have been officially eliminated from the MLS playoff race, with Supporters’ Shield winners FC Cincinnati
dealing the deathblow with a 1-0 victory at DRV PNK Stadium.

The match had every hallmark of what made Inter Miami a flawed side when Lionel Messi arrived. Inter Miami utterly dominated the first half, but could not find
the breakthrough, instead hitting the woodwork three separate times in the opening half-hour and four overall in the match. On the other end, Drake Callender
was nothing short of sensational in goal, but even his heroics were just not enough to stymie the electric counter-attack that FC Cincinnati possess.
</item 4>

<item 5>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Lionel Messi's arrival at Inter Miami has brought with it a craze for fans to get a glimpse of one of the sport's most talented and legendary players of all
time.

Along with that skyrocketing attention comes a leap in prices, and sadly the club seems to be passing that cost on to the supporters.

In 2024, Inter Miami will sport some of the world's most expensive tickets, rivaling the prices of Champions League giants such as Barcelona, Arsenal, and other
massive clubs around Europe, while dwarfing other MLS ticket packages.

While Fox Sports reported the club has an extensive waiting list for 2024 season tickets, the massive increase in price could push out many of the team's most
dedicated supporters who have been with the team since its inception in 2020, causing massive turnover in the fanbase.

A reminder that for most clubs, season-ticket packages include only home league matches, and while members usually get first access to purchase tickets for
knockout competitions, those all come at an additional cost.
</item 5>

<item 6>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Messi, who had missed the last four games due to injury, returned to play 35 minutes off the bench, but by the time he stepped on the pitch, Inter Miami’s early
dominance in the match had come and gone. Messi played a part, but he was unable to fully change the tide of the game.

Eventually, Alvaro Barreal’s 78th-minute goal ended any hope that Inter Miami had to complete a stunning playoff charge, and the defeat coupled with CF
Montreal’s big 4-1 victory over Portland means Inter Miami are officially out of the postseason running.

The Sporting News followed the Inter Miami vs. FC Cincinnati match live, providing score updates, commentary and highlights as they happened.

MORE: Inter Miami playoff scenarios before potential do-or-die match against FC Cincinnati

Inter Miami vs FC Cincinnati final score

Score Goal scorers Inter Miami 0 – FC Cincinnati 1 Barreal (78')

Kickoff: 7:39 p.m.
</item 6>

<item 7>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Code Section Price 2 Mid Loge $45,900 3 North Loge $42,840 4 South Loge $42,840 5 Midfield Club $7,650 6 East Field Seats $13,005 A Northwest Prime $5,440 B
Northwest Midfield $4,420 C West Sideline $2,720 D West Reserved $2,380 E East Midfield $2,635 F East Upper Midfield $2,125 G East Sideline $1,955 H East
Reserved $1,360 I North Goaline $1,054 J South Goaline $1,020 K South Stand $867 L North Stand Reserved $884

Inter Miami season tickets among most expensive in world for 2024

With the Inter Miami season-ticket cost increasing by a huge margin over its own pricing model from 2023 to 2024, the club now boasts some of the most expensive
seats in world football, also rivaling that of other major sports in the U.S.
</item 7>

<item 8>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
A draw or a loss, however, would see them vulnerable to outside results.

If Inter Miami do not win, two of the following three possible outcomes would trigger their elimination from playoff contention: CF Montreal beats Portland
Timbers; NYCFC wins versus D.C. United; and Chicago Fire picks up points against Charlotte FC.

Those are all conceivable results, and therefore Inter Miami will wish to keep it in their own hands just a little longer. They'll eventually need other results
to go their way to climb the table, but for now, their survival is solely up to them.

45 mins to kickoff: Lineups are in, and Lionel Messi is on the bench for Inter Miami.

For FC Cincinnati, Luciano Acosta starts despite the match being meaningless from a tangible standpoint, understandable considering he is chasing the Golden
Boot. Ian Murphy comes in for Yerson Mosquera at the back, while Aaron Boupendza is rotated out up front with Brandon Vazquez and Sergio Santos starting
together up front.
</item 8>

<item 9>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Meanwhile, another nearby rival, Atlanta United, will feature its cheapest season ticket for $605 in 2024, or just $525 for founding members.

Inter Miami season-ticket glitch causes scare

Already, price increases have Inter Miami season ticket holders on edge, but a computer glitch saw supporters fed an even bigger scare.

Mike Ryan, a Miami local who regularly appears on the radio program The Dan LeBatard Show, posted on social media his shock when he received a 2024 renewal for
$30,600 for his two seats, which in 2023 cost just $6,802. Ryan then posted a follow-up a few hours later that stated he was contacted by Inter Miami
representatives who indicated that pricing was incorrect thanks to a computer glitch that sent supporters quotes for additional seats they did not actually
have.
</item 9>

<item 10>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
That's nice and all, but the best way to appreciate the fans would be to not price out the loyal supporters from season tickets next year. Instead, Inter Miami
has jacked 2024 membership rates to put the club amongst some of the most expensive tickets in not just MLS, but the world.

While there will certainly be folks who can afford those high-demand tickets, many of those individuals won't be the same fans who have supported the club from
its inception in 2020. This means Inter Miami have chosen to secure maximum profit at the expense of its fanbase, and a high turnover amongst the regular
in-person support from this season to next is likely.
</item 10>
=== END ITEM id=194f4153bcaadbe1 ===

=== ITEM id=6d4bb1b9fdce403d ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: A year has passed since Altman catapulted ChatGPT to global fame.
  Fact 0.1: Altman has become Silicon Valley's sought-after voice on the promise of artificial intelligence.
  Fact 0.2: Altman has become Silicon Valley's sought-after voice on the potential dangers of artificial intelligence.
  Fact 0.3: Altman's sudden exit was mostly unexplained.
  Fact 0.4: Altman's exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: The piece covers this week's topics.
  Fact 1.1: Sam Altman is backing a teen's AI startup.
  Fact 1.2: Google held a hardware event.
  Fact 1.3: First impressions of the Pixel 8 Pro are covered.
  Fact 1.4: Flexport drama is covered.
  Fact 1.5: The ongoing FTX fallout is covered.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The board's language is a basis for the prevailing theory.
  Fact 2.1: The way giant tech companies work is a basis for the prevailing theory.
  Fact 2.2: The prevailing theory currently floating around is based on the board's language and how giant tech companies work.

Retrieved items:
<item 1>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.
</item 1>

<item 2>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
In a dramatic turn of events late Friday, ex-Y Combinator president Sam Altman was fired as CEO of AI startup OpenAI, the company behind viral AI hits like
ChatGPT, GPT-4 and DALL-E 3, by OpenAI’s board of directors. Then, the company’s longtime president and co-founder, Greg Brockman, resigned — as did three
senior OpenAI researchers. And the fallout continues.

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

It’s a fast-moving situation that we’re still trying to get to the bottom of. No doubt more will become clear as time goes on. To make it easier to follow all
that’s happened in the meantime, though, we’ve put together a timeline; we’ll do our best to keep it current.
</item 2>

<item 3>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Sam Altman has been fired from OpenAI, Inc., the 501(c)(3) nonprofit that acts as the governing body for OpenAI, the AI startup behind ChatGPT, DALL-E 3, GPT-4
and other highly capable generative AI systems. He’ll both leave the company’s board of directors and step down as CEO.

In a post on OpenAI’s official blog, the company writes that Altman’s departure follows a “deliberative review process by the board” that concluded that Altman
“wasn’t consistently candid in his communications” with other board members, “hindering its ability to exercise its responsibilities.”

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

“The board no longer has confidence in [Altman’s] ability to continue leading OpenAI,” the blog post reads.
</item 3>

<item 4>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Any significant snafu in that department would reflect on her, as well as Altman.

5. Perhaps a difference of AI ethics or philosophy?

Altman is a proud techno-optimist, and often speaks fondly of the possibilities of AGI, or artificial general intelligence, a theoretical software system that
achieves human-like intellect and versatility.

The board’s statement pointedly includes that “OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence
benefits all humanity” and that new leadership was necessary. It’s possible that Sam’s zeal for AGI, even absent a secret project or agreement, led to a major
rift between him and the board.

It’s been obvious to all that Altman took the company in a much more corporate direction from its origins, changing its legal status and aggressively pursuing
enterprise and consumer applications. That doesn’t sound a lot like the “mission” the board wants to advance.
</item 4>

<item 5>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 5>

<item 6>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 6>

<item 7>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 7>

<item 8>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 8>

<item 9>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 9>

<item 10>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 10>
=== END ITEM id=6d4bb1b9fdce403d ===

=== ITEM id=566f9a73d011e187 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that 'all of that, all of it, was built on lies.'
Sentence 1: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 1.0: Bankman-Fried used Caroline Ellison as a front.
  Fact 1.1: Caroline Ellison is Bankman-Fried's on-again off-again girlfriend.
  Fact 1.2: Caroline Ellison is the CEO of Alameda.
  Fact 1.3: The government claimed Bankman-Fried had 'secret access' to customer money.
  Fact 1.4: The government claimed the customer money was both cash and crypto.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth.
  Fact 2.2: The prosecution said Bankman-Fried committed fraud to achieve great power.
  Fact 2.3: The prosecution said Bankman-Fried committed fraud to achieve great influence.
  Fact 2.4: The defense countered that the FTX founder acted in good faith.
  Fact 2.5: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.6: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 4>

<item 5>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 5>

<item 6>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.
</item 8>

<item 9>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 9>

<item 10>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 10>
=== END ITEM id=566f9a73d011e187 ===

=== ITEM id=59ace4c1d84f0a1a ===
Gold sentences and facts:
Sentence 0: Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to keep their hands off of some social media posts.
  Fact 0.0: The Supreme Court decided last week that it would hear the pair of cases.
  Fact 0.1: The pair of cases revolve around Republicans crafting state-specific laws.
  Fact 0.2: The state-specific laws order platforms to keep their hands off of some social media posts.
Sentence 1: Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation.
  Fact 1.0: Under Musk, Twitter has prioritized free speech.
  Fact 1.1: Twitter is now called X.
  Fact 1.2: Under Musk, free speech on Twitter is managed by crowdsourced-based moderation.

Retrieved items:
<item 1>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“Thanks to the First Amendment, Florida and Texas cannot force websites or social media apps to host hateful content, misinformation and spam, as their deeply
misguided laws would require,” Oregon Senator Ron Wyden, who co-authored Section 230, a law that protects social media companies’ content moderation decisions,
told TechCrunch. “… A ruling in favor of the Texas and Florida laws would create utter chaos, and make many sites worthless to regular users who want to watch a
funny video or see family photos.”

NetChoice President Steve DelBianco also warned that allowing the state laws to go into effect would unleash “a tidal wave of offensive content and hate speech
crashing onto users, creators, and advertisers” that would force Americans to wade through “racial epithets, aggressive homophobia, pornographic material,
beheadings, or other gruesome content” just to use social apps.
</item 1>

<item 2>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“They’re in the same ballpark, the sentiment is the same,” Barrett said. “Republican lawmakers in each state believe that — and said explicitly in the course of
debating and passing these laws — that ‘Silicon Valley oligarchs’… are ideological liberals, and they are censoring people in our states who are conservative,
and we are hereby ordering them to stop doing that.”

How could the Supreme Court decision affect social media companies?

If the court finds that social media companies don’t have a First Amendment right to curate the kinds of content that they allow, social platforms could look
very different, at least in states that are trying to limit their moderation powers. After years of slow progress on misinformation — and worrisome backsliding
on platforms like Elon Musk’s X — the Supreme Court’s decision could upend that process, sowing chaos online in the process.
</item 2>

<item 3>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What does this have to do with the First Amendment?

The case revolves around First Amendment rights — but, counterintuitively, it’s the rights of social media companies that are in question, not the rights of
their users.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment?”

“These cases are about the First Amendment and how the First Amendment applies to social media companies. And then more specifically, what the First Amendment
has to say about content moderation, which is obviously a subset, although a really big and important subset of what social media companies do,” Barrett said.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment — or what you also
might call content moderation — in sorting out what expression does and does not appear on the platforms that they own? So it’s just not clear right now what
precisely the First Amendment says about that question.”

Are the Texas and Florida laws identical?
</item 3>

<item 4>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
To explain the Supreme Court’s decision to wade into these issues — and what happens next — TechCrunch spoke to Paul Barrett, NYU adjunct law professor and
Deputy Director of NYU Stern’s Center for Business and Human Rights.

Why is the Supreme Court involved?

These cases actually started in Florida and Texas a few years ago before wending their way to the Supreme Court this year. In those two states, Republican
lawmakers passed parallel laws to control how social media companies operate. In Florida, Governor Ron DeSantis signed Senate Bill 7072 into law in May 2021. In
Texas, House Bill 20 made it through the state legislature and was signed by Governor Greg Abbott in September 2021.

“The reason why these cases are in front of the Supreme Court is actually relatively simple: Florida and Texas were more or less the first out of the gate in
imposing this type of restriction on social media companies,” Barrett explained.
</item 4>

<item 5>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“So when the industry sued the states under the First Amendment, these were the first cases that were litigated, so they went up through the court system.”

Both laws made their way through the lower courts after tech industry group NetChoice and the Computer and Communications Industry Association (CCIA) issued
legal challenges against them. That path was complicated and contradictory, which is part of how the case landed in the Supreme Court’s lap:

In both cases, you had federal trial judges who entered injunctions blocking the laws on constitutional grounds. And then you had two different federal
appellate courts — in the case of Florida, the Eleventh Circuit, in the case of Texas, the Fifth Circuit. The two appellate courts clashed… and there was an
explicit conflict between the two federal appellate courts. And that type of conflict is one of the bases that the US Supreme Court uses for deciding when to
take cases.

What does this have to do with the First Amendment?
</item 5>

<item 6>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 6>

<item 7>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation. Its policies have resulted in increased hate speech
and toxic content on the platform, some studies have shown.

Mozilla believes there’s a better way forward and, more importantly, that now is exactly the right time to invest in that path. And, as a wholly owned
subsidiary of a nonprofit, the company says it’s not motivated by generating earnings for shareholders or returning a VC investment, allowing it to progress
with a collaborative approach where it takes in input from a lot of different voices.

In an interview with TechCrunch, Mozilla Senior Director of Content Carolyn O’Hara explained why Mozilla has taken an interest in the fediverse and Mastodon,
specifically, and how it expects to experiment in the months ahead.
</item 7>

<item 8>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What’s at stake in the Supreme Court’s landmark social media case The court will wade into First Amendment questions with major implications for platforms

The Supreme Court is poised to decide whether a pair of state laws are allowed to reshape the ability of social media companies to control what does — and
doesn’t — appear on their platforms.

Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to
keep their hands off of some social media posts. Since the early days of the Trump administration, Republicans have accused social media companies of
deliberately suppressing conservative viewpoints.

While research has not supported these claims, researchers have demonstrated that conservative social media users are disproportionately exposed to political
misinformation, a phenomenon that could explain anecdotal claims of ideologically lopsided enforcement on social platforms.

Whether animated by those perceived differences or the political perks of accusing social platforms of anti-conservative bias, conservative lawmakers in Florida
and Texas passed laws to restrict how those companies are allowed to moderate content.
</item 8>

<item 9>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Aside from forcing platforms to allow some forms of content that would otherwise be disallowed, these laws also seek to force social media companies to provide
users with individualized explanations when their content is removed or restricted. Because this process is largely conducted algorithmically now — generally
with light human intervention or oversight — social media companies might need to reimagine their content moderation systems or hire way more humans to respond
to these incidents. Those kind of adjustments would likely be costly and difficult to scale.

“As you can imagine, that can get kind of onerous when you’re taking down millions and millions of pieces of content a day, and much of that activity of the
vast majority of it is currently being done automatically,” Barrett said. “The idea that a human being would have to go back and explain each time something
came down would be quite a challenge.”
</item 9>

<item 10>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 10>
=== END ITEM id=59ace4c1d84f0a1a ===

=== ITEM id=251506431803d90a ===
Gold sentences and facts:
Sentence 0: Here's your chance to claim $5,000+ in bonuses as a new user from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, and BetRivers.
  Fact 0.0: The reader has a chance to claim $5,000+ in bonuses as a new user.
  Fact 0.1: A new user can claim bonuses from Caesars Sportsbook.
  Fact 0.2: A new user can claim bonuses from BetMGM.
  Fact 0.3: A new user can claim bonuses from Bet365.
  Fact 0.4: A new user can claim bonuses from FanDuel.
  Fact 0.5: A new user can claim bonuses from DraftKings.
  Fact 0.6: A new user can claim bonuses from BetRivers.
Sentence 1: Collect your welcome bonus: As a new customer, you’ll receive a welcome bonus.
  Fact 1.0: The reader can collect a welcome bonus.
  Fact 1.1: A new customer receives a welcome bonus.

Retrieved items:
<item 1>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
You need to win the wager to receive $150 in bonus bets.

Bet now: FanDuel promo code for Bet $5 on any moneyline, Get $150 in bonus bets if your team wins

DraftKings promo code scores $1,200 in bonus bets for Tennessee vs. North Carolina odds

Use our DraftKings promo code link to unlock $1,200 in bonus bets for Tennessee vs. North Carolina! DraftKings offers new users a 20% deposit match up to $1000,
$50 Bonus Bet with $5 deposit + Bet $5, Get $150 in Bonus Bets promo. All customers will also receive a "No Sweat SGP" daily!

Registering at DraftKings is simple, and so is claiming $1,200 in bonus bets.
</item 1>

<item 2>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Use Bet365 bonus code VATSN in Virginia or NJTSN in New Jersey.

Claim here: Bet365 bonus code TSNCO for promo of your choice

FanDuel promo code for Bet any $5 moneyline, Get $150 in bonus bets if your team wins in CBK game

FanDuel has a new offer when using our exclusive FanDuel promo code link. Bet $5 on any moneyline, and Get $150 in bonus bets if your team wins!

It's important to note that there are no odds restrictions, so it'd be prudent to bet a big favorite over an outmatched underdog on the CBK schedule. You need
to win the wager to receive $150 in bonus bets.
</item 2>

<item 3>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Keep in mind that when you wager with Bonus Bets (aka DK Dollars) and win, you only receive the winnings and not the wager amounts.

DraftKings Sportsbook welcome offer pros and cons

Pros

There are multiple welcome offers and not just a single offer

You don’t have to risk a lot — just $5 — to receive a total of $200 in Bonus Bets

You also don’t have to worry about winning or losing your first wager to get a bonus as with other sites’ offers

Cons

The 20% deposit match bonus worth up to $1,000 might sound good, but with a 25x wagering requirement it is truthfully only useful to high-stakes bettors

You only get a week to use the $150 in Bonus Bets awarded for making your first wager of at least $5

You only receive winnings and not the bet amounts when you use Bonus Bets

What will the welcome offer at DraftKings Sportsbook VT be?
</item 3>

<item 4>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
DraftKings | BetMGM vs. FanDuel | Caesars vs. BetMGM
</item 4>

<item 5>
College Bowl Odds, Spreads, Money Lines and Totals - NCAAF 2023 Bowl Season (Sporting News, 2023-12-15T20:37:00+00:00)
It’s a quick process. Below, we outline the steps for opening your own account.

Choose your sportsbook: If you’re not sure which sportsbook is the right fit for you, browse our reviews to see the pros and cons of each, as well as the
current offerings for new customers. When you find the sportsbook that best suits your needs, simply click on a link from our site to visit the homepage of your
chosen sportsbook. Register for an account: On the homepage, you’ll find a button that says “sign up” or “register.” Once you click that, you’ll be taken to a
form for providing personal information such as your name, birth date, email address, and the last four digits of your Social Security number. You’ll also be
required to accept the sportsbook terms and conditions, which we recommend you read. Collect your welcome bonus: As a new customer, you’ll receive a welcome
bonus.
</item 5>

<item 6>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking on our exclusive DraftKings promo code link

Register as a new user by supplying personal info including your name, email, last four of your SSN#, and mailing address

Deposit $5 to activate $50 bonus bet

Bet $5 and Get $150 in bonus bets

To max out DraftKings' deposit match at $1,000, make a minimum deposit of $5,000 and meet the 25X playthrough requirement.

Claim here: DraftKings promo code for $1,200 in bonus bets

BetRivers promo code SPORTSRIV unlocks 2nd chance bet for CBK games including Duke vs. Arkansas

The BetRivers promo code SPORTSRIV gives new users a 2nd chance bet up to $500 depending on your state.

If your first bet settles as a loss, BetRivers will refund your account in bonus bets.
</item 6>

<item 7>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
into your FanDuel account and place a qualifying . Whether your bet wins or loses, you receive $200 in Bonus Bets .

. You can use your Bonus Bets in whatever increments you like, but you have to use your Bonus Bets within seven days or they will expire.

or they will expire. Winning wagers with Bonus Bets award the winnings only and not the bet amounts.
</item 7>

<item 8>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
The promo codes on this page get you the best bonuses available. When you plug in the code during registration, the sportsbook knows we sent you and gives you
the bonus we negotiated for you.

Are these promo codes for regular customers, too?

The promo codes on this page are for new customers only. Once you’ve got an account with a sportsbook, you can take advantage of other bonus offers and ongoing
promos.

Related Pages:
</item 8>

<item 9>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Here are more details on the offer explaining how each part of the DraftKings Sportsbook welcome offer works:

Sign up for a DraftKings Sportsbook account via the links on this page .

. Make a first deposit of at least $5 and get a $50 Bonus Bet .

and get a . When you make your first deposit, DraftKings will give you a 20% deposit match bonus worth up to $1,000 in Bonus Bets . Be aware this bonus comes
with a 25x wagering requirement .

worth up to . Be aware this bonus comes with a . Finally, when you place an initial bet of at least $5 , you get $150 more in Bonus Bets delivered as eight $25
Bonus Bets that you’ll have seven days to use.

, you get delivered as eight $25 Bonus Bets that you’ll have seven days to use. If you go for the full deposit match bonus, you can receive as much as $1,200 in
Bonus Bets all told.
</item 9>

<item 10>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
This offer requires a minimum $10 deposit, but you can deposit up to the amount of your first bet to $1,500 to maximize your potential winnings. If your first
bet on the app loses, BetMGM will refund your account with bonus bets in the amount of the original wager up to $1,500.

The extra $50 bonus bet offer is not available in Washington D.C., Kentucky, Mississippi, Nevada, New York, Puerto Rico and Ontario. Customers in those areas
can use BetMGM bonus code SPORTSPICK for the $1,500 First Bet offer.
</item 10>
=== END ITEM id=251506431803d90a ===
