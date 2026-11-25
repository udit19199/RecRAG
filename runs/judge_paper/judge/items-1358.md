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

This file has 1 items.

=== ITEM id=3a1bc38256baf93f ===
Gold sentences and facts:
Sentence 0: In the third quarter, Uber generated $394 million in operating income and $219 million worth of net income.
  Fact 0.0: In the third quarter, Uber generated $394 million in operating income.
  Fact 0.1: In the third quarter, Uber generated $219 million worth of net income.
Sentence 1: The ride-hail company claims that the rate of sexual assault reported on the app decreased 38% between its first and second reports.
  Fact 1.0: The ride-hail company claims that the rate of sexual assault reported on the app decreased 38%.
  Fact 1.1: The 38% decrease was between the ride-hail company's first and second reports.

Retrieved items:
<item 1>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 1>

<item 2>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
A Google spokesperson said the ability to see live traffic conditions and business information was halted temporarily out of “consideration for the safety of
local communities.” Google did something similar in 2022 amid the Russian invasion of Ukraine because the apps were being used to track military movements.

XPeng, often called the Chinese challenger to Tesla, has removed high-definition mapping in its XNGP assisted driving feature following its rival’s lead.

Ride-hailing and car-sharing

Rapido, the eight-year-old Indian bike taxi startup, is expanding into the cab market in the South Asian nation, where Uber and its homegrown competitor Ola
dominate.

Peer-to-peer car-sharing marketplace Turo has introduced a buy-now-pay-later option. Now you can reserve a car without immediately paying for it until seven
days before a trip. Turo says it built the BNPL option in-house.

UAW strike

Progress was made as the United Autoworkers strike wrapped up its sixth week.
</item 2>

<item 3>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
This comes just a few months after the company announced a significant wave of layoffs.

May 2023

Taxfix

Announced on May 30 that it has laid off 20% of staff—120 employees.

Meta

Announced May 24 that it is laying off about 6,000 people. In total, about 21,000 people have lost their jobs at Meta since November.

JioMart

Reliance Retail’s online shopping platform laid off over 1,000 employees on May 22, and plans to cut as many as 9,900 more roles over the coming weeks.

Krebs Stamos Group

Announced May 18 that the consulting firm laid off six people. In April, the firm had 18 employees including the founders. On May 18, the site shows only 14
team members.

TuSimple

Announced May 18 that it is laying off about 30% of employees.

Prior to the layoff, TuSimple had about 550 employees in the U.S.
</item 3>

<item 4>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Groupon

Impacting another 500 employees announced on January 31. The company said this new set of layoffs will be spread across the first two quarters of 2023.

Impossible Foods

Reportedly affecting 20% of its staff, over 100 employees, Bloomberg reported first.

PayPal

Announced on January 30, about 2,000 full-time employees, or 7% of its workforce, were affected.

Arrival

Announced on January 30, with a newly appointed CEO, slashing 50% of its workforce — 800 employees globally.

Waymo

The self-driving technology unit under Alphabet quietly laid off workers on January 24, according to The Information and several posts on LinkedIn and Blind.
It’s not yet clear how many of Waymo’s staff will be affected.

Spotify

Announced on January 23, impacting around 6% of its global workforce — around 600 employees.
</item 4>

<item 5>
The new, ‘efficient’ Spotify has a very different approach to podcasting (The Verge, 2023-10-25T16:10:29+00:00)
CEO Daniel Ek echoed that sentiment. “We’re constantly finding new ways to bring more efficiencies out of the business… We’ve seen some improvements, but you
should expect us to continue to look for more improvements going forward because that’s just our modus operandi.”

While Ek and Vogel really hammered that “efficiency” theme (according to my transcript, they said the word more than 30 times over the course of the 50-minute
call), what was left unsaid about podcasting spoke louder. In the past, they would excitedly tout how many millions of podcasts were on the platform, the latest
new celebrity show, and the area’s explosive growth. But as investors have run out of patience for the time and money such ambitions required, the company’s
leadership changed its tune.

Rather than emphasizing, say, its new revenue sharing deal with Trevor Noah or the reliability of The Ringer, Ek pointed to Spotify’s new AI-driven translation
product.
</item 5>

<item 6>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
That decision was surprising to me based on how sources had described an all-hands meeting earlier in the week that was led by co-founder and CEO Kyle Vogt. In
that meeting, which came the day after the DMV suspended Cruise’s permit, Vogt and other leaders told staff the company had not paused operations elsewhere
besides California and gave no indication that the company was planning to. Instead, Vogt told employees the company was re-evaluating how it discloses
information to regulators to ensure it is clearly communicated, according to an account from sources who heard the call.

Cruise had even quietly launched driverless operations in Miami (just a few vehicles), a move that suggested the company was moving ahead despite its
significant problems in California.

What changed? Perhaps Cruise execs were pressured by GM or they looked around and realized that they were losing support from other states.
</item 6>

<item 7>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Mojocare

Announced on June 18 that the company will be laying off some 150 employees, a majority of the workforce.

Tada

Operated by Korean ride-sharing company Socar’s subsidiary VCNC announced on June 15 that it plans to let go of at least 50% of its staff or about 45 people at
Tada.

Better.com

Announced on June 7 that it laid off its real estate team, it is not clear how many people were impacted.

Ursa Major

At least 14 people have been laid off from the rocket engine startup, announced on June 7. Ursa Major did not respond to TechCrunch’s inquiry into the layoffs,
or the number of people affected. According to LinkedIn, 292 people are listed as currently working at Ursa.

Spotify

Announced June 5 that it will cut 200 jobs in its podcast unit, leading to a workforce reduction of 2%. This comes just a few months after the company announced
a significant wave of layoffs.
</item 7>

<item 8>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
In-vehicle cameras as a deterrent

Abrams cited data from studies of taxis equipped with in-vehicle cameras, which she says has drastically reduced instances of sexual assaults against
passengers, as well as assault of passengers against drivers.

“These are predatory, opportunistic drivers taking advantage of vulnerable women, so if they’re on film, they likely wouldn’t commit the crime,” Abrams told
TechCrunch.

Uber didn’t respond to TechCrunch’s request for more information about why it hasn’t mandated the use of cameras in ride-hail vehicles, but as with any
surveillance, there are privacy issues. The legality of mandating cameras also varies across local and state laws.

Abrams has her own theory as to why Uber has been slow to implement security cameras.

“Cost isn’t the issue,” she said. “It’s that it would deter drivers because a lot of drivers don’t want cameras. And so if they don’t have drivers, they don’t
make money.”
</item 8>

<item 9>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Of course, adjusted EBITDA is to profit as hidden heels in men’s shoes are to height, so we’ll want some harder figures as well. In the third quarter, Uber
generated $394 million in operating income and $219 million worth of net income. Mix in $966 million worth of positive operating cash flow, and Uber looks very
healthy, given that all of the figures in this paragraph were improvements on its year-ago results.

But while food delivery and scooting humans about town were profitable for Uber in the quarter, the final major portion of its business had a more lackluster
quarter.

Freight falls

One area that continues to drag on Uber is its freight business.

While Uber’s ride-hailing and delivery business saw an uptick in gross bookings in the third quarter, Uber Freight experienced a 27% drop year-over-year.
Revenue, as a result, had a similar fall.

The business unit reported revenue of $1.3 billion in the third quarter, a 27% drop from the same period last year.
</item 9>

<item 10>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Judge Charles Breyer in the Northern District of California will preside over pretrial hearings. This will be the first time a federal judge is able to make
decisions for a large amount of these cases, which will streamline the proceedings. Breyer may preside over the trials depending on whether the parties agree.
If the parties do not agree to have Judge Breyer preside of the case for trial, the cases go back to their home jurisdiction for trial.

Another consolidated lawsuit has also been filed against Uber in California, but it only covers survivors in that state.

Uber has attempted, through several filings for motions to dismiss, to stop the consolidation of these cases. It argues that the company did not owe a duty to
the plaintiffs to protect against criminal conduct. Lawyers representing the survivors will have to prove that Uber owed a duty of care to passengers, including
the duty of taking reasonable precautions to ensure their safety.

The pretrial matters, including witness and expert depositions and document discovery, will be heard by Judge Charles Breyer.
</item 10>

<item 11>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
So far, its plan to forgo the gigafactory for local microfactories has proved anything but, writes Harri — thanks to missed production targets, low cash
reserves, layoffs and a pivot.

Audio

It’s winter, it ain’t getting warmer (at least here in NYC), and I’d argue that there’s no better place to be than snuggled up indoors with a podcast for
company. If you’re in need of material, TechCrunch has a few that should definitely be on your radar.

This week on Equity, the crew dove deep into the encouraging signs from the fintech startup market, starting with Klarna’s Q3 results. From there, they looked
at buy now, pay later consumer behavior and fintech fundraising results with a 2021 flavor.

Meanwhile, Found featured Nasrat Khalid of Aseel, which started as an e-commerce company making it possible for local artisans in Afghanistan to sell to
customers across the world.
</item 11>

<item 12>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Third-party Reddit app Narwhal says it hopes to survive Reddit’s app purge with a subscription plan.

Apropos “getting together,” Amanda reports that VidCon is still kickin’. For the first time, the conference hosted an industry leadership summit, where creator
economy experts and creators could hash out their grievances with the state of the business and share ideas to make the job of a creator more sustainable. That
makes sense, unlike creators raising venture capital: It’s so eye-wateringly hard to make money as a creator, I’d love to see the pitch that convinces a VC to
cut a check to a creator, with a realistic expectation of a venture-scale return.

Maybe they were hiding behind the sofa?: Sarah reports that Mastodon actually has 407K+ more monthly users than it thought — and it seems like Twitter has a lot
more traffic than Musk said. Still, the peak now is about the same as it was a decade ago, and it’s unclear what the social platform can do to encourage more
growth.
</item 12>

<item 13>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Future of flight

EVTOL company Archer Aviation plans to start air taxi operations in Abu Dhabi in 2026. From there, Archer says it will expand across the UAE as part of a
memorandum of understanding with the Abu Dhabi Investment Office.

Miscellaneous

The United Auto Workers strike is affecting CES. Stellantis canceled its planned presentations for the tech trade show in January, citing the cost of ongoing
UAW strikes.

Zipcar is getting hit with a fine from the NHTSA for “renting vehicles with open, unrepaired recalls.” One recall concerns 2015–2017 Ford Transit Vans, which
continued to appear on Zipcar’s platform despite safety issues with the vehicles.

People

Autobrains hired Uri Yacovy, a former SVP at Mobileye, as its chief operating officer.

Logistics company Flexport is laying off 20% of workers, or about 600 people, topping off a spate of staff upheavals at the company.
</item 13>

<item 14>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Snap

Laid off fewer than 20 project managers November 8. CNBC reported that Snap’s vice president of engineering, ​​Nima Khajehnouri, will also be leaving.

Amazon

Started cutting workers in its music streaming division in Latin America, North America and Europe, the company confirmed November 8.

Google

Confirmed November 8 it eliminated “a small number of roles” in its consumer support staff, Google Users & Products.

Virgin Galactic

Said in a memo to staff November 8 it is cutting costs and reducing staff. The number of employees impacted is currently unknown.

Ubisoft Montreal

Confirmed November 7 it will cut 98 roles in business administration and IT. Ubisoft said 124 positions will be eliminated overall across Canada, including
reductions in its global IT team and its VFX studio, Hybride.

Nextdoor

Revealed in its Q3 earnings that it is reducing its workforce by 25%.

Beyond Meat

Announced cuts November 6 affecting around 19% of its non-production employees.
</item 14>

<item 15>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.
</item 15>

<item 16>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The combined impact of those two items was 8 percentage points of growth.

The profit perspective

From bookings to revenues to profits: How did the main portions of Uber’s business generate black ink?

Turning to what Uber calls “segment adjusted EBITDA,” it’s not hard to see how the company managed to rack up more income in its most recent quarter.
Ride-hailing adjusted profit rose to $1.29 billion, up 43% from $898 million in the year-ago quarter, while delivery saw its own profitability skyrocket from
$181 million in Q3 2022 to $413 million in its most recent fiscal period.

Uber did spend more during Q3 2023 when compared to its Q3 2022 result, but the 5% gain in “Corporate G&A and Platform R&D” costs to $595 million was far less
than the gains we saw above. So, Uber’s adjusted EBITDA rose from $516 million in aggregate to $1.09 billion in the third quarter of this year.
</item 16>

<item 17>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The latest funding round brings QI Tech’s total capital raised to $262 million.

New Web Summit CEO appointed: Former Wikimedia Foundation CEO Katherine Maher has been appointed as the new CEO of Web Summit, following the controversial
departure of its previous leader, Paddy Cosgrave. Maher’s appointment comes as the tech conference faces a challenge to rebuild relationships and trust within
the tech community.

Top reads on TechCrunch this week

Shinier, smarter maps: Google is rolling out a variety of updates for Maps, incorporating AI technology to enhance user experience. The feature is currently
available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.

From stop-and-start to charging ahead: Over on TC+, Tim has had it with U.S. automakers hesitating on EVs, as Ford and GM are showing signs of backing away from
their commitment to electric vehicles. Cowards.
</item 17>

<item 18>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Anyone spotting a trend here?

Nikola, the electric and hydrogen-powered heavy truck maker, was awarded $165 million from its founder and former executive chairman, Trevor Milton, in an
arbitration proceeding.

Nio has opened its 2,000th Power Swap Station in China, nearing its goal to build 2,300 stations by the end of 2023. The company has expanded on its strategy of
swapping out EV batteries, rather than charging them, an infrastructure-intensive process that has the potential to make topping up a battery as quick as
filling up a gas tank.

Tesla has the attention of the U.S. Department of Justice — again. This time it has received requests for information, including subpoenas from the DOJ related
to perks, the advertised range of its EVs and personnel decisions.

In-car and mobile tech

Google Maps and Waze stopped live traffic updates in Israel and the Gaza Strip at the request of the Israel Defense Forces.
</item 18>

<item 19>
Why Gogoro picked India as its new go-to market (TechCrunch, 2023-12-19T04:10:35+00:00)
The company is not limited to two-wheelers in its home country but also offers its swapping-battery technology to players that run auto-rickshaws.

Nonetheless, the growth of Gogoro is stagnant and is even dropping in Taiwan, as Luke admitted to TechCrunch.

In its recent earnings report, the Nasdaq-listed company mentioned a 10.2% year-on-year drop in its revenue of $91.8 million, resulting in a net loss of $3.1
million, down from a net income of $56.4 million in the same quarter last year.

He underlined there were a couple of reasons for seeing a decline in its financials and dropping its business in Taiwan. First, he said, it was due to a
lobbying effort against the speed of electric vehicle adoption after the 2020 elections. Second, the country took some time to return from the COVID-19
pandemic.

“Taiwan has always been a pilot for us,” Luke stated.
</item 19>

<item 20>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Reminder that you can drop us a note at tips@techcrunch.com. If you prefer to remain anonymous, click here to contact us, which includes SecureDrop
(instructions here) and various encrypted messaging apps.

Micromobbin’

This week was the trade show Micromobility America. Here are some of the best bits that came out of it:

Bird is now doing e-bikes again? But instead of the VanMoof knockoff of previous years, Bird has partnered with TradeHubb, an e-bike supplier, and Spring, a
strategic retail growth company, to launch the bike. You might recall that Bird last year ditched its retail bike offering to focus on shared rides in an
attempt to reach profitability. The struggling company was also recently delisted from the stock market. Who knows what’s going on behind the scenes? I tried
reaching out to learn more, but no response. If you have a tip, hit me up!
</item 20>

<item 21>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Aside from in-vehicle surveillance and more extensive background checks, the survivors are asking Uber to implement driver training on interactions with
passengers, a zero-tolerance policy for drivers, sexual harassment education and training, and a more adequate system to encourage customer reporting and
monitor customer complaints.

Are Uber’s in-app safety features working?

Abrams argues that Uber’s safety features are insufficient because they’re all app-based and only potentially useful if a rider has access to their own phone.

“The majority of clients I represent and the women I’ve spoken to who have been sexually assaulted have experienced: (A) someone else ordered the ride for them;
(B) their phone is dead or they can’t locate their phone to use the safety shield; or (C) if they could use their phone, they’re incapacitated,” said Abrams.

The lawyer says she has interviewed upward of 5,000 survivors over the course of several years and has seen no difference in the number of attacks before or
after implementation of these in-app features.
</item 21>

<item 22>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
: Shield AI, a startup specializing in autonomous flying systems for the U.S. military, has raised $200 million in a new funding round, reaching a valuation of
$2.7 billion.

Eero grows up, with best-in-class Wi-Fi: Amazon’s Eero has launched its most advanced mesh Wi-Fi router yet, the Eero Max 7.

Boksing claver: Danish startup Soundboks released its fourth-generation Bluetooth speaker that has been a hit in Europe’s festival scene but has yet to gain a
strong foothold in the U.S. market.

Highs and lows in the world of startups

The current bear market in cloud stocks has led to a reversal of revenue multiples from earlier this year, making it difficult for late-stage startups to accrue
value, Alex writes on TC+. However, early-stage startups that are growing rapidly may still be able to raise funds and demand for a better valuation. The trend
points toward a favorable environment for startups prioritizing growth over cash preservation.
</item 22>

<item 23>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
I think the leadership at Hyundai — the chairman visited me last week, and we’re in close touch with him and the people who work with him — think that getting
into the 22nd century is an important thing to do and we’re getting started on it now.”

Pratt adds, “There’s actually this alignment that’s happening between cars and robotics, as well. Rod Brooks was the one who a few years ago said that modern
cars are eldercare robots. I think that’s really true. You think about the amount of computers in them, the amount of software that’s in them. TRI has different
divisions inside of it, one of which works on very advanced things in cars that is completely overlapping with some of the software and some of the concepts
that we’re using in the robotics space.”

For its part, the Boston Dynamics AI Institute lists three key pillars of its research: intelligence, dexterity and mobility. That’s effectively making robots
that are smarter, better at mobile manipulation and move more dynamically.
</item 23>

<item 24>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Welcome to Startups Weekly. Sign up here to get it in your inbox every Friday.

A few months ago, Alexis Ohanian, the co-founder of Reddit, tweeted that if he could go back in time and do one thing differently when he was building Reddit,
he would have spent significantly less time attending events. Personally I have a different experience — I always regret getting on a plane (or in an Uber, for
that matter) to go to an event. However, as I’m traveling back, I’ve never regretted it: There’s always something magical that comes out of going to an event,
in my experience.

Our TC+ team is not interested in anecdotes, however: neither mine nor Alexis’. So, like the data-driven journalists they are, they surveyed more than 50
founders, trying to figure out whether attending events still makes sense in 2023.

Spoiler alert: There isn’t a real consensus, but there are some really good pros and cons.
</item 24>

<item 25>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Multiverse

Will lay off nearly a third of their U.S. staff. It’s the second round of layoffs at the U.K.-based unicorn company in just over a month.

ONE

Is cutting its workforce by around 25% November 27, citing “market conditions” affecting the EV battery startup.

ByteDance

Started mass layoffs November 27 in its gaming department, Nuverse, after two years of tepid performance. The number of affected employees is unknown.

Veev

Is closing up shop, according to multiple reports, after reaching unicorn status in 2022.

Tulip

Laid off 25 employees, CEO Ali Asaria announced November 24.

Anar

Is shutting down operations and returning capital to investors, CEO Nishank Jain announced November 23.

Alerzo

Cut 100 workers November 21, reportedly due to increases in automation. The recent layoffs come after Alzero cut 15% of its workforce earlier this year and 5%
in 2022.
</item 25>

<item 26>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The trend points toward a favorable environment for startups prioritizing growth over cash preservation.

At the same time, we wouldn’t be surprised if WeWork ends up filing for bankruptcy (TC+). The company has been grappling with significant challenges. WeWork’s
stock hit a new low, dropping over 47% after hours to just $1.21, which brings the company’s market cap down to a mere $121 million. This is a drastic fall from
the $47 billion valuation it achieved after its SoftBank-led Series H round in early 2019.

Hope for the best, prepare for the worst: Startup co-founders don’t always make it, and in her fantastic TC+ piece, Rebecca explores what you can do to prepare
today, should a co-founder breakup be on your startup journey’s path.

Bumper banking backing in Brazil: Brazilian startup QI Tech, a banking-as-a-service platform, has raised $200 million in a Series B funding round, marking the
largest raise in Brazil this year across all sectors.
</item 26>

<item 27>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 27>

<item 28>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
Three years later, Pratt was named TRI’s CEO. Raibert continued as Boston Dynamics’ CEO until 2019, handing the reins over to longtime employee Rob Playter, who
managed the company’s shift to more commercial pursuits. Last year, Raibert (who still serves as BD’s chairman) founded the Boston Dynamics AI Institute. The
organization has a lot in common with TRI. Both are dedicated to pure research, with the backing of two major automakers (Toyota and Hyundai).

I recognize that “pure research” is a bit of a loaded term. The concept is certainly a hopeful one: finding a method that can sustain research outside of the
pressures of academia and corporate R&D.

Asked whether there is pressure from Toyota to productize, Pratt says, “There actually is not, and I’m not saying that just to be nice. I think one of the great
things about companies that have existed for long times is that this isn’t the first time that they’ve done R&D.
</item 28>

<item 29>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
The new round values the Bengaluru-headquartered electric vehicle startup at about $5.4 billion

Pony.ai, the Chinese autonomous vehicle startup, scored $100 million from Neom, Saudi Arabia’s futuristic city and development project. As part of the deal, a
joint venture will be established to develop, manufacture and deploy autonomous vehicles and smart infrastructure in Neom and key markets in the Middle East
North Africa region.

Neom also announced plans this week to set up a $10 billion joint venture with Danish freight forwarder DSV.

Stellantis made a €1.5 billion deal ($1.59 billion) to take a 20% stake in Chinese electric vehicle maker Zhejiang Leapmotor Technologies, just days after
ending manufacturing in the country. The deal includes the formation of Leapmotor International, a 51% to 49% Stellantis-led joint venture that has exclusive
rights for the export and sale, as well as manufacturing, of Leapmotor products outside China.
</item 29>

<item 30>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 30>

<item 31>
European neobroker Scalable Capital raises $65M on a flat $1.4B valuation (TechCrunch, 2023-12-07T07:29:44+00:00)
(More on that theme in the latest annual, barometric report from Atomico.)

“Yes, we are four or five times bigger than we were at the last round,” Podzuweit said in an interview. “So like for like, obviously it’s a very attractive deal
for the investor. But in the current market environment, I think it’s really cool that we managed to get to pull this off.” He confirmed that the company still
has “the majority of the money from the last funding round” in the bank, but this extra injection means it can be a little more aggressive in its approach. “We
can do a bit more marketing, we can hire the top people but we also have a much bigger equity cushion,” he said. The company has now raised more than $345
million, it said.

Indeed, the fundraise may be coming at a tight time for startup finance, but one likely reason that Scalable took the money now is that, for neobrokers, the
landscape is getting more competitive and crowded.
</item 31>

<item 32>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Crunchbase

Announced on LinkedIn on July 9 that the company is laying off 52 employees.

Microsoft

Announced July 10 that the company is eliminating additional jobs a week after the start of its 2023 fiscal year. These layoffs are in addition to the 10,000
layoffs announced in January.

ClickUp

Announced on July 4 that it has laid off 10% of its workforce, about 90 people.

June 2023

Niantic

Announced June 29 that it laid of 230 employees — one year after it laid off around 90 employees.

Plex

Announced June 29 that it laid off approximately 20% of its staff, or 37 employees.

Grab

Announced the week of June 23 that it laid off more than a thousand people, or 11% of its staff.

Olx Group

Announced on June 20 that it has cut around 800 jobs globally. The news comes as the company started to close operations of its automotive business nit Olx
Autos in some markets.
</item 32>

<item 33>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Hundreds of women have filed lawsuits against Uber claiming the company hasn’t done enough to prevent instances of sexual assault by drivers. Now a panel of
judges has ruled that more than 80 cases can be consolidated into federal court.

The stakes are high for both parties with implications extending to future Uber riders and drivers. The outcome of the case could result in sweeping changes to
Uber’s platform, which plaintiffs argue will reduce sexual assault and also raise new concerns over privacy.

The upshot? Technology lies at the center of this very human story.

Uber has attempted to address sexual assaults by drivers — which the lawsuits claim Uber has known about since 2014 — through new safety features in its app,
like a 911 button and the ability to share location with a friend. Yet survivors, and their attorneys, say that response has been inadequate, and they’re
calling for better technological solutions like in-vehicle surveillance cameras.
</item 33>

<item 34>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
The catch? GPS and other electronic devices are strictly prohibited.

So what am I checking out at this seemingly non-tech event? EVs and tech, of course. Oh, and green hydrogen, believe it or not.

There were 10 vehicles out of the 65 that fell into the electrified category such as the Jeep Wrangler Rubicon 4xe. Four of those vehicles — all of them Rivian
R1T pickups — were electric. And this year, one Rivian team took first place in the 4×4 class (there are two classes in the Rebelle) — the first time an
all-electric vehicle was on the top podium. The first place finishers, driver Lillian Macaruso and navigator Alexandra Anderson, are both employees of Rivian.

OK, Rebecca, back to you.

Tesla reported its third-quarter earnings this week, and once again, all eyes were on the automaker’s margins amid ongoing price cuts.
</item 34>

<item 35>
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
</item 35>

<item 36>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.
</item 36>

<item 37>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Deal of the week

Kirsten here again! Convoy isn’t a traditional deal of the week, but its collapse sure got my attention.

The digital freight broker told employees this past week it was shutting down due to what executives described as a “massive freight recession.” It turns out
that disrupting the freight business is hard.

The abrupt closure, which wiped out investors, comes about 18 months since the Seattle-based company raised $260 million in fresh funding that pushed its
valuation to $3.8 billion.

I went back and read an interview TechCrunch conducted with Convoy co-founder and CEO Dan Lewis. A few things he said stuck out, namely what led him to start
the company in the first place.
</item 37>

<item 38>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Superpedestrian

Is shutting down its U.S.-based shared scooter operation and exploring a sale of its European business, TechCrunch has exclusively learned.

Bolt

Confirmed it laid off 29% of its staff December 15. The e-commerce company known for its one-click checkout technology was at one time the subject of a federal
probe.

Cruise

In an announcement obtained exclusively by TechCrunch, the embattled self-driving car company is cutting 900 employees, or 24% of its workforce, per a December
14 email from new president and CTO Mo Elshenawy.

Etsy

Is laying off 11% of its workforce. In a letter to employees December 13, CEO Josh Silverman cited a “very challenging macro and competitive environment.”

Chipper Cash

Has reportedly laid off 15 people across various departments. It would be the fintech unicorn’s fourth round of layoffs this year.

SmileDirectClub

Has shut down three months after filing for Chapter 11 bankruptcy, citing $900 million of debt.
</item 38>

<item 39>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
VinFast

VinFast has not shared how many employees have been cut, but a LinkedIn post from a former employee said “nearly 35 roles” were affected. Announced on February
6.

Dell

Announced February 6, impacting 6,650 people, or 5% of worldwide workforce.

Getaround

Announced February 2, 10% of staff — about 42 employees.

Pinterest

Announced February 2, 150 employees impacted. This is the second job-cutting move within weeks of the first round in December 2022.

Rivian

Announced on February 1, cutting 6% of its workforce for the second time in less than a year.

January 2023

SoFi Technologies

Announced on January 31, cutting 65 jobs, or about 5% of its 1,300-person workforce. First reported by The Wall Street Journal.

NetApp

Announced on January 31, impacting 8% of its staff — about 960 people.
</item 39>

<item 40>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
And so if they don’t have drivers, they don’t make money.”

Many drivers install their own dash cams to record trips, usually as backup evidence for insurance claims or to defend themselves against unfair deactivations
from Uber’s platform. Uber is also piloting a new video recording feature for drivers that allows them to record video on their smartphones. But in those cases,
the driver can decide what and when to record, and when to share that data.

Other demands

The survivors in the joined lawsuit also allege that Uber’s “fast and shallow background checks” are substandard and designed to make it as easy as possible for
drivers to sign up quickly. Uber uses third-party companies like Checkr and Appriss to do background checks, which Sergio Avedian, senior contributor at The
Rideshare Guy, says “are at best watered down and not guaranteed of bad apples from falling through the cracks.”

The lawsuit calls on Uber to also include fingerprinting, which would run prospective drivers through FBI databases.
</item 40>

<item 41>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Either way, Cruise said it’s now going to examine “processes, systems, and tools and reflect on how we can better operate in a way that will earn public trust.”

That might be a hefty challenge, especially in California. As the Cruise drama unfolded, opposition against robotaxis grew in cities like Los Angeles. And two
of the biggest groups to oppose robotaxi expansion in California are now formally working together.

Want to reach out with a tip, comment or complaint? Email Kirsten at kirsten.korosec@techcrunch.com or Rebecca at rebecca.techcrunch@gmail.com.

Reminder that you can drop us a note at tips@techcrunch.com. If you prefer to remain anonymous, click here to contact us, which includes SecureDrop
(instructions here) and various encrypted messaging apps.

Micromobbin’

Taiwanese battery swapping giant Gogoro came to play at the Japan Mobility Show 2023, showing off how its scooter batteries can also be used to power a tiny
car.
</item 41>

<item 42>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Aside from the 911 button and location-sharing feature, which were both introduced in 2018, Uber has added other features to the app over the last five years.
In 2021, Uber introduced a feature that allows riders and drivers to record audio during a trip. The following year, Uber launched a pilot to provide passengers
with live help from an ADT safety agent, as well as PIN verifications to ensure passengers are connected to the correct driver.

Rachel Abrams — a sexual assault attorney and partner at law firm Peiffer Wolf Carr Kane Conway & Wise, one of the law firms working on the multidistrict
litigation, who filed the petition to consolidate the various actions in federal court — argues the app-based solutions are half-measures that haven’t quelled
instances of sexual violence against passengers on the platform. Mandatory in-vehicle cameras in Uber cars is “essential for safety,” she says.

Uber told TechCrunch it cannot comment on pending litigation but said it remains committed to the safety of all users on its platform.
</item 42>

<item 43>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Together the plan is to deploy more than 1 million chargers globally.

Volta Trucks filed for bankruptcy proceedings in Sweden as difficulties with suppliers proved a hindrance to raising funds. Volta said the bankruptcy in August
of Proterra, an EV parts supplier, and the uncertainty over its own battery supplier means it needed to cut down the number of trucks it could produce. Volta,
which is based in Sweden but has operations in the U.K., also said it would file for bankruptcy in Britain.

Notable reads and other tidbits

Autonomous vehicles

Cruise, General Motors and Honda are launching a robotaxi service in Japan under a new joint venture. The service will launch with Origin vehicles in Tokyo in
2026.

Speaking of Cruise, the National Highway Traffic Safety Administration (NHTSA) has opened an investigation into the GM subsidiary’s AV system following several
incidents involving pedestrians in San Francisco.
</item 43>

<item 44>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
After all, that would require Gogoro to set up a swapping network in those regions. But if so, it would signal a new revenue stream for the company, which has
been struggling to reach profitability amid softening demand and large investments into international expansion.

— Rebecca Bellan

Deal of the week

Well this is a fun one.

Flexport is in talks to acquire the technology of Convoy, the once buzzy digital freight startup that abruptly shuttered after failing to find a buyer. This
possible deal, which was reported by WSJ, didn’t have any other details, but it still made me raise an eyebrow.

Lest you forget, Flexport founder Ryan Petersen just took back the CEO title after his hand-picked successor was pushed out. Petersen’s big message has been
getting the company’s financial house back in order and has criticized former CEO Dave Clark of overspending, specifically around hiring and expanding too
quickly.
</item 44>

<item 45>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
2. Do they disagree on long-term strategy?

Despite being the hottest tech company in the world right now and everyone talking about ChatGPT, OpenAI isn’t exactly a sound business. It’s shoveling money
into the furnace as fast as it can by serving, by all accounts, a fantastically expensive product at bargain-bin prices.

That’s all well and good for a year or two, but at some point that strategy changes from a growth hack to an existential liability. Could Altman and the board
have had irreconcilable differences on where that point lies?

This doesn’t seem so likely. The company has been very deliberately pursuing this very publicly, confidently and on a long-term basis. Altman and the board seem
to be in sync on this, at least for the present.

3. Do the numbers not add up?

On the other hand, could OpenAI be losing even more money than Altman admitted or projected? It seems impossible, but the costs of running this operation have
no precedent, nor really does the operation itself.
</item 45>

<item 46>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Tesla’s shares, which are priced more as a tech stock than as an automaker’s stock, were down after Q3 earnings. Investors were clearly feeling skittish after
Tesla reported a gross margin of 17.9%, down from 25.1% in the same period last year. That’s also down from Q2’s margins of 18.2%. As a result, profits fell 44%
to $1.85 billion in Q3 from the same year-ago period.

Investors see the falling margins and Tesla’s price cuts as proof that demand is lessening for the vehicles as other EVs take market share and rising interest
rates make it difficult for many buyers to afford big ticket purchases. The company also reported that solar deployments slipped 48% in Q3 from the same period
last year. But the company made up for it by pulling in a 90% spike in energy storage deployments.

Tesla also gave some updates about its long-delayed Cybertruck.
</item 46>

<item 47>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 47>

<item 48>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
The former Amazon and Google executive, who has a background in strategy and management consulting, told TechCrunch that when he was struck by the urge to start
a company, he researched the money-attracting industries of the world, and then, using AngelList, saw how many companies were trying to disrupt those
industries.

Here’s the nugget:

His search yielded thousands of companies that were working on industries ranging from telecommunications and fashion to video games and food. Billions of
dollars were going into trucking each year but fewer than 30 startups showed an interest in the field.

“I saw a massive opportunity and few people going after it,” Lewis told TechCrunch.

And then later, when asked if his method of deciding on a direction for a startup is still a good method, Lewis said, in short, yes. Read the whole interview
from May 2022 here.

Other deals that got our attention . . .
</item 48>

<item 49>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 49>

<item 50>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Synapse

Confirmed October 6 that it had laid off 86 people, about 40% of the a16z-backed fintech company. Earlier in the year, Synapse laid off 18% of its employees
after its growth slowed.

Bird

Issued a new round of layoffs after acquiring Spin, according to an email interim CEO Michael Washinushi sent to the company. The layoffs came one week after
the company was delisted from the New York Stock Exchange, and the exact number remains unknown.

Qualtrics

Announced on October 4 that 780 roles are being eliminated and that “several hundred” roles are changing or moving locations in 2024.

Hopper

Is reducing its full-time staff by 30%, amounting to 250 job cuts, the company announced October 3.

Chainalysis

Announced October 2 that it is laying off around 15% of its workforce, affecting approximately 150 employees.
</item 50>

<item 51>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 51>

<item 52>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
The number of incidents increased in the categories of nonconsensual kissing, nonconsensual touching and nonconsensual penetration, or rape, between 2017 and
2019.

The total number of rape incidents also rose from 2017 to 2019. And while the number of reported rapes dropped in 2020, the rate has remained the same since
2017. In other words, there has not been an improvement in the rate of rapes on the platform.

Sexual assaults are an ongoing problem for Uber

Uber has been sued numerous times over the past few years by passengers who claim they were sexually assaulted during a ride. Lyft has faced similar lawsuits
and accusations.

Under Uber’s terms of use, class action lawsuits can’t be filed against the company in cases of sexual assault, so each case has to be heard individually. This
has prevented survivors from advocating for themselves collectively.

Judge Charles Breyer in the Northern District of California will preside over pretrial hearings.
</item 52>

<item 53>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Lyft

Announced on April 21 that employees will learn whether they have a job or not via an email that will be sent out April 27.

Lyft layoffs to affect 26% of workforce, or about 1,072 people as promised on April 27.

Meta

Announced April 18th that it is expected to lay off 10,000 jobs in the coming months. This is on top of the 11,000 jobs that were cut in November.

Redfin

Announced April 13th that it has laid off 201 employees, about 4% of its workforce. This is the third time the Seattle-based real estate company has reduced its
workforce since June.

Apple

Bloomberg reported on April 3 that Apple is laying off a small number of roles on its corporate retail teams.

March 2023

Netflix

Announced on March 31, Netflix confirms a “handful of layoffs,” which includes two longtime executives. The exact number of layoffs is unclear.
</item 53>

<item 54>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Despite cloud computing remaining critical, AI budgets are now more than double that of cloud computing. Eighty-two percent of business leaders believe the
integration of AI solutions will increase their employee performance and job satisfaction, and startups should expect a high level of desire for and experience
with AI solutions in their future customers.

Finally, we’ve seen consumption, or usage-based priced tech products’ growth slow in recent quarters. Will that fact lead startups building modern AI tools to
pursue more traditional SaaS pricing? (The OpenAI pricing schema based on tokens and usage led us to this question.)

The trajectory of usage-based pricing has organically aligned with the needs of large language models, given that there is significant variation in
prompt/output sizes and resource utilization per user. OpenAI itself racks upward of $700,000 per day on compute, so to achieve profitability, these operation
costs need to be allocated effectively.

Nevertheless, we’ve seen the sentiment that tying all costs to volume is generally unpopular with end users, who prefer predictable systems that allow them to
budget more effectively.
</item 54>

<item 55>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
The Bengaluru-headquartered startup employed about 150 people at peak and raised more than $130 million over its eight-year journey.

Audio

TechCrunch’s roster of podcast episodes keeps growing — just in time for weekend listening.

Equity featured a throwback conversation from TechCrunch Disrupt 2023, when Alex sat down with Serhii Bohoslovskyi, the founder of a no-code app builder,
Trible, that helps people construct online courses. The pair caught up on the state of the creator economy, the use of no-code tooling today (and how it’s
received by nontechnical creators) and the security of startups with roots in Ukraine.

Over on Found, the crew talked to David Rogier, the CEO and founder of MasterClass, a streaming platform where you can learn from the world’s experts on a range
of topics.
</item 55>

<item 56>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Notable reads and other tidbits

Autonomous vehicles

Waymo driverless vehicles are now available through the Uber app, starting with Phoenix. The launch comes five months since the two companies announced a
multi-year agreement for the autonomous vehicle service to be accessed via the Uber app.

Electric vehicles, batteries & charging

Ford is delaying about $12 billion in planned investments on EVs, including construction of a second battery plant with joint venture partner SK On due to
softening demand for higher-priced premium electric vehicles. While EV sales have grown, consumers aren’t willing to pay a premium for an EV over a gas or
hybrid vehicle. That price pressure has squeezed profits, and in the case of Ford’s EV business caused losses to grow.

General Motors and long-time partner Honda have ended plans to build millions of affordable and smaller electric vehicles as the automakers come to terms with
high interest rates and battery costs coupled with softening EV demand. Anyone spotting a trend here?
</item 56>

<item 57>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.
</item 57>

<item 58>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
account-to-account payment grows

Recapitalization, $60M Series D support growth of e-commerce financier Clearco

Stitch raises $25M Series A extension led by Ribbit Capital, increasing the round’s total to $46M

Resy and Eater co-founder raises $24M for Blackbird, a restaurant loyalty platform

Sparx wants to do for enterprise what Truebill did for consumer recurring bills

Ten Key Labs wants to simplify managing equity for startups

Seen elsewhere

Logistics company Loop raises $35M to modernize the supply chain (Also, check out TechCrunch’s previous story on Loop.)

Shift4 acquires SpotOn unit for $100M (TechCrunch previously covered SpotOn here.)

Reserv raises $20M for AI-driven insurtech software

Vyzer raises $6.3M seed round for AI-powered wealth management platform

Kafene adds another $12.6M to its Series B round (TechCrunch previously covered Kafene here.)
</item 58>

<item 59>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
Certainly most of the R&D budget inside of Toyota is spent figuring out how to make the next car or maybe the car five years from now. But there’s this notion —
and I’m sure Hyundai shares this — that we’re at the once-in-a-century time for transformation in the car industry. Battery electric vehicles are much simpler
to make than cars have been in the past — no engine, no transmission, etc. So, we’re going to have to compete in a much fiercer way in cars. But also, can we
use the skills, the dreams and the hopes of the company to go beyond cars? TRI’s job is to actually think about [what’s] next. What’s next after cars, as well
as some fancy stuff on cars.”

For TRI, much of the “what’s next” focuses on supporting aging populations. The organization invests a good portion of its resources to building out technology
designed to help older people live more independently. That’s at the root of the research we shared from the institute last week.
</item 59>

<item 60>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
The lawsuit calls on Uber to also include fingerprinting, which would run prospective drivers through FBI databases.

“This is a very intentional and deliberate decision as evidenced by Uber’s active lobbying and resistance against municipalities and regulatory bodies
implementing any kind of biometric fingerprinting requirements for drivers,” said Kevin Conway, managing partner at Peiffer Wolf Carr Kane Conway & Wise.

Uber has lobbied against additional background requirements for drivers, which has given the company authority to conduct its own background checks with little
or no oversight, unlike most taxi operators.

Uber and Lyft say smudged fingerprints can lead to inaccurate results and that fingerprint checks reference historical arrest records, which can have
discriminatory effects on some minority communities that face disproportionately high arrest rates. An Uber spokesperson told CNN arrest records are incomplete
and often lack information about whether a person has been convicted of a crime.
</item 60>

<item 61>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
The idea, O’Hara said, is to rethink social networking from the ground up.

Ford shutters SaaS app for field work: Ford has shut down VIIZR, a software-as-a-service company that, along with Salesforce, built an app to help tradespeople
like plumbers, locksmiths and electricians to schedule field appointments, send invoices and manage customers, Kirsten reports. VIIZR, which was announced in
December 2021, was a separate company majority owned by Ford, with Salesforce as a minority investor.

Apple bets on generative AI: Apple CEO Tim Cook pushed back against the notion that the company was behind in AI on Apple’s Q4 earnings call with investors, as
he highlighted technology developments that Apple had made recently that “would not be possible without AI.” Cook also said that Apple was working on generative
AI technologies, lending credence to reports suggesting the company is on track to spend $1 billion per year on developing generative AI products.
</item 61>

<item 62>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 62>

<item 63>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
PayPal’s fees are, however, driven by the costs imposed upon them by the dominant card schemes, Visa, Mastercard and American Express. So, we can expect PayPal
will respond to this suit by arguing that it is as much of a price taker as the merchants themselves, and that the terms imposed upon it by the card schemes
drive some of its anti-steering rules.”

PayPal did not respond to requests for comment. — Mary Ann

Bolt CEO gets frank about SEC probe

Bolt Financial CEO Maju Kuruvilla told me in an interview that the one-click checkout company is putting a recent U.S. Securities and Exchange Commission probe
behind it and is moving on.

“It obviously is a very, very involved process, but we’re really happy to put that behind us,” Kuruvilla told TechCrunch.
</item 63>

<item 64>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Any other metric could be wrong and misleading.

For example, price by revenue or price by EBITDA is incorrect. Why? How do you differentiate between two companies A and B — both having same market cap of ₹100
and similar in all means except for debt levels — same revenue (₹100) and same gross margin and same EBITDA margins of 20 per cent and thus same EBITDA of ₹20.
But A has zero debt, while B has debt of ₹10. Since revenue or sales and EBITDA are same, if you do a market cap/revenue or Price /Sales, or marketcap/EBITDA,
both companies will have the same valuation.

However, let’s assume depreciation is ₹10, interest cost is 10 per cent and taxes are zero. This means the net profit of A is ₹10, while that of B is ₹9.
</item 64>

<item 65>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Other deals that got our attention . . .

Hayden AI, an AI and geospatial analytics company, raised $53 million in a Series B funding round led by the Drawdown Fund. The company’s tech is being used by
government agencies to enforce traffic violations that obstruct transit buses and capture data to help increase ridership and improve traffic efficiency.

Commercial fleet insurance startup Nirvana Insurance has raised a $57 million Series B to expand its big data platform, hire new staff and grow its business in
the trucking industry. Lightspeed Venture Partners led the round, with General Catalyst and Valor Equity Partners also participating. The round doubles
Nirvana’s valuation to more than $350 million post-money.

Supply chain logistics company Transfix raised a $40 million Series F. The company is backed by New Enterprise Associates, G Squared and Canvas Ventures.

EV charging company Wallbox has acquired the operations and assets of German-based EV charging solutions company ABL for €15 million.
</item 65>

<item 66>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
February 2023

Twitter

Announced on February 26, the company laid off more than 200 employees, including Esther Crawford, Haraldur Thorleifsson and Leah Culver. Since Musk took over
Twitter in October last year, the company’s headcount has fallen by more than 70%.

Poshmark

Announced February 24, Poshmark confirmed with TechCrunch that less than 2% of its workforce was affected, primarily in the U.S. The company employs roughly
800+ employees.

Green Labs

We do not have an exact figure of how many Green Labs plans to lay off its staff. Green Labs confirmed to TechCrunch that it is conducting a round of layoffs
that could impact at least 50% of its workforce.

Chipper Cash

Announced on February 20, the African cross-border payments platform conducted a second round of layoffs just 10 weeks after it cut approximately 12.5% of its
workforce. Chipper Cash relieved almost one-third of its workforce, about 100 employees.
</item 66>

<item 67>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 67>

<item 68>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 68>

<item 69>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The question is whether Uber believes in the long-term income potential of freight.

So what?

When discussing Uber’s results internally, our vibe was that it is a very profitable and healthy company today, albeit one that is not growing as fast as the
market had hoped. The fact that Uber’s share price is up as we write this indicates that investors are content thus far to excuse the small Q3 misses and keep
their eyes more focused on its year-over-year improvements and forward guidance. More when we get Lyft’s numbers.
</item 69>

<item 70>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
as an important early step in its plans for an initial public offering, a Klarna spokesperson tells TechCrunch+.

The unicorn’s legacy isn’t over: It’s been 10 years since Cowboy Ventures’ founder Aileen Lee coined an incredibly catchy nickname for what were very rare
startups at the time: Unicorns. TechCrunch+ spoke with Lee about how she feels about the term 10 years later, now that her venture firm is also a decade old.
</item 70>

<item 71>
The small and affordable EVs we lost in 2023 (TechCrunch, 2023-12-19T15:08:55+00:00)
VanMoof proclaimed at the time that it was the “most funded e-bike company in the world,” and co-founder Taco Carlier boasted the company was “reinventing,
redesigning, and re-engineering every component of the bicycle.”

VanMoof’s rapid growth and focus on proprietary parts led to its downfall. By 2023, the company reportedly lost money on each bike due to costly repairs, and
struggled to raise additional funds to stay afloat. This past summer, VanMoof paused sales and Taco Carlier left as the firm declared bankruptcy.

Electric scooter maker Lavoie acquired what was left of VanMoof in August for “tens of millions,” pulling it out of bankruptcy. Months later, it seems the
acquirer has yet to restart VanMoof production or resume repairs.

On the other hand

It’s been a bummer year for fans of small EVs, but all is not lost. For one, city bike-sharing programs are booming and increasingly electric. D.C.
</item 71>

<item 72>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 72>

<item 73>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Tesla also gave some updates about its long-delayed Cybertruck. Initial deliveries are set for an event at Giga Texas on November 30. Elon Musk noted that
scaling the Cybertruck will be hard and it will take 18 months before the pickup is profitable.

“I mean, we dug our own grave with Cybertruck,” said Musk.

The billionaire executive also said Giga Texas will be able to produce about 250,000 Cybertrucks a year starting in 2025. But let’s remember that Musk isn’t
great at making predictions. After all, he initially said the Cybertruck would be on the market by 2021. Expect some of these numbers to be pushed out, too.

Want to reach out with a tip, comment or complaint? Email Kirsten at kirsten.korosec@techcrunch.com or Rebecca at rebecca.techcrunch@gmail.com.

Reminder that you can drop us a note at tips@techcrunch.com.
</item 73>

<item 74>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
Cowards.

A swan song from the bird sanctuary: X has marked its value at $19 billion, according to internal documents, marking a 56% decrease in value since Elon Musk’s
purchase last year. Bad for shareholders and Musk, but potentially good for executives and employees, Amanda explains.
</item 74>

<item 75>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
The Station is a weekly newsletter dedicated to all things transportation. Sign up here — just click The Station — to receive the newsletter every weekend in
your inbox. Subscribe for free.

Welcome back to The Station, your central hub for all past, present and future means of moving people and packages from Point A to Point B.

It was an absolutely wild week on the robotaxi front, and more specifically for GM’s self-driving car subsidiary Cruise.

The week started off with the California Department of Motor Vehicles suspending Cruise’s driverless and deployment permits (with the California Public
Utilities Commission following shortly after), effectively ending the company’s robotaxi operations in San Francisco just months after receiving the last
necessary permit to commercialize its operations.

Two days later, Cruise decided to pause driverless operations in every market it had started to charge for its robotaxi service, including Austin, Houston and
Phoenix.
</item 75>

<item 76>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 76>

<item 77>
Twitch’s money guy talks about the revenue split controversy and its monetization long game (TechCrunch, 2023-10-26T22:28:10+00:00)
I have to wonder if those decisions to change these policies were in response to competition from other platforms.

I would say it this way: I think they are an acknowledgement, and maybe a closer connectivity to the community via Dan specifically. I credit Dan in his short
time as CEO to really make some big, hard decisions. Both the launch of Partner Plus, as well as simulcasting, the embed stuff we talked about in the keynote —
these are three things that I think are very much an acknowledgement of the feedback from the community. We’ve made that change, so I wouldn’t call it out of
competitive pressures. I would call it out of focus on the community.
</item 77>

<item 78>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A comprehensive list of 2023 tech layoffs From major layoffs at Google, Amazon and Microsoft to small fintech startups and apps

Last year’s techwide reckoning continues. The tech industry has seen more than 240,000 jobs lost in 2023, a total that’s already 50% higher than last year and
growing. Earlier this year, mass workforce reductions were driven by the biggest names in tech like Google, Amazon, Microsoft, Yahoo, Meta and Zoom. Startups
across many sectors also announced cutbacks through the first half of the year. And while tech layoffs slowed down in the summer and fall, it appears that cuts
are ramping up yet again.

Many economists have cautioned against fears of a recession, which would seem like a reason for optimism. But the momentum for a tech sector rebound has been
slow to build, resulting in tech companies continuing to cut back on their workforces and pivot from a growth mindset to one based on efficiency in the face of
stubborn market conditions.
</item 78>

<item 79>
What’s up with Tesla’s Cybertruck? Everything to know about much-hyped electric pickup (TechCrunch, 2023-11-30T12:05:46+00:00)
What’s up with Tesla’s Cybertruck? Everything to know about the much-hyped electric pickup

After four years, the long-awaited launch of the Tesla Cybertruck electric pickup has come and gone.

The boxy vehicle is Tesla’s first new model since 2020, when it started delivering the Model Y. Yet, Cybertruck’s initial debut predates that moment; Tesla CEO
Elon Musk showed off an early version of the pickup at a memorable 2019 event, when it accidentally smashed two windows while attempting to demonstrate
Cybertruck’s durability.

Here we’ll answer some questions we figured a brave explorer such as yourself might ask (err… type in a search bar) about Tesla’s Cybertruck, including details
on the vehicle’s specs, availability and design of the vehicle. The first deliveries, in which about 10 high-profile customers like Reddit co-founder Alexis
Ohanian took possession of the truck, occurred November 30 at the Tesla Gigafactory in Austin.
</item 79>

<item 80>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Microsoft has poured $13 billion into OpenAI so far and has what amounts to a 49% stake in the company. The former’s stock price fell more than 1% in the last
30 minutes of trading, after Altman’s departure was announced; Axios reports Microsoft was notified Altman would be leaving “minutes” before the public
announcement.

OpenAI was founded as a nonprofit in 2015, but restructured in 2019 as a “capped-profit” company to raise capital — an acknowledgement of the huge costs
associated with training cutting-edge AI systems. Underlining the point, Altman said in an interview this week with the Financial Times that he “hoped”
Microsoft would increase its investment to help pay for “huge” imminent model training costs.

When contacted via email, OpenAI’s press relations declined to comment beyond the official blog post.
</item 80>

<item 81>
The new, ‘efficient’ Spotify has a very different approach to podcasting (The Verge, 2023-10-25T16:10:29+00:00)
This is Hot Pod, The Verge’s newsletter about podcasting and the audio industry. Sign up here for more.

In pursuit of profit, Spotify looks to AI rather than original content for its podcasting future

Spotify shareholders are thrilled with the company reporting an operating profit for the first time in a year, sending the stock up nearly 10 percent on the
news. The revenue bump was in large part due to the streamer’s $1 price increase earlier this year, but executives also pointed to the downsizing of the podcast
operation — which included hundreds of layoffs and the dissolution of Gimlet and Parcast — as a contributing factor. The adage that “content is king” no longer
applies, with Spotify increasingly focusing on tools that will scale the business rather than content that will attract listeners.

In a call with investors, CFO Paul Vogel said that after being a real “drag” on profit margins in the past, podcasting will soon break even and turn toward
profitability. CEO Daniel Ek echoed that sentiment.
</item 81>

<item 82>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The company is now facing an unfair labor practice complaint, with workers alleging the layoffs are retaliation for unionizing.

Chewy

Laid off more than 200 employees November 14. The cuts impact roles in HR, recruiting, data, business intelligence, plus directors and higher managers.

Amazon

Is cutting “just over 180” roles in its gaming division, VP Christoph Hartmann wrote in an email to employees November 13.

Carta

Is reportedly cutting more of its workforce. The number of positions affected are unknown, but it would be the unicorn startup’s third layoff round this year.

Pico

Is cutting a “few hundred” roles, a person familiar with the matter told TechCrunch November 9. ByteDance’s Oculus challenger held an internal meeting
announcing it is halting its expansion this week.

Cruise

Started laying off contingent workers who supported driverless operations November 9. Those layoffs came after the company recalled 950 robotaxis and lost key
commercial permits to operate in California.
</item 82>

<item 83>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
It has evolved into working in humanitarian aid, delivering emergency food supplies to people in need in Afghanistan and Turkey.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Another superconductor disappointment: Tim writes that a new, supposedly room-temperature superconducting material isn’t what the scientific community hoped it
would be. With the Nature-published paper detailing the material facing retraction, the odds of researchers discovering a room-temperature superconductor are
looking even longer.

Klarna inches toward an IPO: Mary Ann and Alex write that Swedish fintech Klarna is taking steps toward an eventual IPO. The company has initiated a process for
a legal entity restructuring to set up a holding company in the U.K.
</item 83>

<item 84>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
If you’re not, consider signing up. Here are a few highlights from this week:

Bitcoin surge: Jacquelyn writes about Bitcoin’s rapid-fire ascent to $44,000, which came on the back of roughly 25% gains in the last week. Her piece for TC+
explores what’s driving Bitcoin’s price ascent and similar value gains among other tokens — and whether the good vibes continue into the new year.

To swap, or not to swap: Tim reports on how consumer EV battery swapping could usher in freedom for a wide range of people, allowing them to participate in the
EV transition in ways that traditional built-in batteries don’t. The challenge is making the unit economics work.

Coinbase and Robin and the future of fintech: Investors are betting that consumer trading of equity and crypto is rebounding and are consequently pushing the
value of some former startups higher, Alex writes. That could spell good news for startups offering consumer trading services directly — or indirectly, for that
matter.
</item 84>

<item 85>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
In a note to employees Monday, Spotify founder and chief executive Daniel Ek — citing slow economic growth and rising capital costs — said right-sizing the
workforce is crucial for the company to face the “challenges ahead.”

TuSimple exits: When TuSimple went public in 2021, it was flying high as the leading self-driving trucks developer in the U.S. Now — after a string of internal
controversies and the loss of a critical partnership with truck manufacturer Navistar — TuSimple is exiting the U.S. altogether. TuSimple said in a regulatory
filing Monday that it’s laying off the majority of its U.S. workforce and selling assets here as it exits the country for Asia.

ZestMoney shuts down: ZestMoney — a buy now, pay later startup whose ability to underwrite small-ticket loans to first-time internet customers attracted many
high-profile investors, including Goldman Sachs — is shutting down following unsuccessful efforts to find a buyer.
</item 85>

<item 86>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Prior to the layoff, TuSimple had about 550 employees in the U.S. and post reduction in force the company will have about 220, according to the company.

Nuro

Announced May 12 that it will lay off 30%, or about 340 employees across the company.

LinkedIn

Announced May 8 that it is cutting 716 jobs, or about 3.6% of total employees, and will phase out its local jobs app in China. Despite reducing some roles,
LinkedIn also plans to open about 250 new jobs on May 15.

Rapid

Announced May 5 that Rapid (previously known as RapidAPI) has laid off another 70 employees less than two weeks after letting go of 50% of its staff. Just 42
people remain at the company, down from 230 in April, dropping a total of 82% in headcount.

Meesho

Announced May 4 that it has cut 15% of its workforce, or 251 roles.
</item 86>

<item 87>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Petersen has spent the past month cutting costs, including laying off about 20% of its workers, or about 600 people.

If Flexport acquires the technology, the company plans to restore Convoy’s trucking services for as many customers and partners as possible, according to WSJ’s
source. And folks, that’s going to cost money. Is Petersen’s reign of financial frugality already over?

Other deals that got my attention …

Faction, the driverless tech developer, raised an undisclosed amount in a round led by TDK Ventures. Ducera Partners, Trucks Venture Capital and Fifty Years
also joined the round.

Ola Electric raised $384.4 million in a funding round, which included about $240 million in debt. Singapore’s sovereign wealth fund Temasek led the funding
round and Indian government-backed lender State Bank of India bankrolled the debt.
</item 87>

<item 88>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
You cannot escape conversations about AI no matter how far or fast you run. Hyperbole abounds around what current AI tech will be able to do (revolutionize
every industry!) and what current AI tech will be able to do (take over the world!). Closer to the ground, TechCrunch+ is working to understand where startups
might find footholds in the market by levering large language models (LLMs), a recent and impactful new method of creating artificially intelligent software.

How AI will play in startup land is not a new topic of conversation. A few years back, one venture firm asked how AI-focused startups would monetize and whether
they would suffer from impaired margins due to costs relating to running models on behalf of customers. That conversation died down, only to come roaring back
in recent quarters as it became clear that while LLM technology is quickly advancing, it’s hardly cheap to run in its present form.

But costs are only one area where we have unanswered questions.
</item 88>

<item 89>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.
</item 89>

<item 90>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
“We look forward to focusing on the momentum for the business and how we can help the retailers, especially the big retailers who are looking to us to innovate
for them, because this is a tough year for retailers.”

You might remember that Bolt, which provides checkout technology to merchants, and its co-founder Ryan Breslow, were subpoenaed last year by the SEC to
investigate whether the company violated any securities laws during fundraising in 2021 when Bolt was seeking its $355 million Series E round that valued the
company at $11 billion.

The investigation took about 15 months, but news of that probe didn’t go public until July of 2023. Shortly thereafter on August 23, the SEC said, in a letter
viewed by TechCrunch, it was not recommending an enforcement action for the company.

Kuruvilla spoke with me about how, exactly, the company can put something like this behind it, what it told customers and what’s next. Read more.
</item 90>

<item 91>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Then a bigger one. Then an effectively limitless one.

Ellison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have believed you if you told me I would
be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more comfortable with.”

Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When FTX took a loss in
the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s “insurance fund,” meant to
keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.

Even that wasn’t good enough.
</item 91>

<item 92>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
13; Sardine, No. 16; and Tapcheck, No. 48.

CB Insights also released its Fintech 100 list, made up of the “most promising 100 fintech startups of 2023.”

Other items we are paying attention to:

The great Zelle pool scam

The Public platform offers everyday investors a slice of ‘Shrek’ catalog with quarterly payouts

Stockpile and Green Dot partner on debit cards for minors

Amex pilots biometrics in online checkout

Pie Insurance appoints Audra Foglietta as chief financial officer

Monzo picks Cash App vet Conor Walsh for US CEO

Folsom-based EV Life launches electric car loan to lower monthly payments

CLEAR launches reusable KYC solution

Funding and M&A

As seen in TechCrunch

Open banking led to a fintech boom — as Brite raises $60M, account-to-account payment grows

Recapitalization,
</item 92>

<item 93>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.
</item 93>

<item 94>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Domo

Made reductions to 7% of staff, in addition to reductions in its contract workforce, CEO Joshua G. James announced in the company’s Q3 earnings call November
30.

Mojo

Is laying off 20% of its staff, the company announced to staff November 30. The A-Rod and Marc Lore-founded sports betting startup previously raised $100
million.

Loco

Is cutting its workforce by about 36%, co-founders Anirudh Pandita and Ashwin Suresh announced at a November 30 town hall.

Unity

Is laying off 265 workers November 29 after eliminating its Wētā Digital division, which it acquired in 2021.

Tier Mobility

Is cutting roughly 22% of its staff, CEO and co-founder Lawrence Leuschner announced November 28.

Dataminr

Is laying off about 20% of its staff starting November 28, citing “the recent rapid advancements of our AI platform,” according to a memo from founder and CEO
Ted Bailey.
</item 94>

<item 95>
European neobroker Scalable Capital raises $65M on a flat $1.4B valuation (TechCrunch, 2023-12-07T07:29:44+00:00)
We’re hopefully hearing from Yared directly later today and will update this more after that.

Updated to correct the amount raised in 2021 to $180 million, not $140 million; and to specify the number of customers and the total raised, after the numbers
were provided by the company post-publication.
</item 95>

<item 96>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
Hey, folks, and welcome to Week in Review (WiR), TechCrunch’s newsletter covering the past week (or so) in tech industry happenings. This week marked OpenAI’s
first-ever dev conference, where the Microsoft-backed AI startup announced a host of new products. But that was far from the only item of note.

In this edition of WiR, we spotlight Brian’s review of the 16-inch M3 Max MacBook Air and M3 iMac 24-inch; Mozilla betting on a decentralized social networking
future; Ford shuttering a company that was building an app for plumbers, electricians and other trades; and Tim Cook’s thoughts on generative AI. Also on the
agenda is WeWork officially filing for bankruptcy, Bumble getting a new CEO, and the spectacular failure of EV startup Arrival.

It’s a lot to get through, as always — so we won’t delay.
</item 96>

<item 97>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 97>

<item 98>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
On a quarter-over-quarter basis, Uber Freight had a 1% gain in revenue.

The results don’t get any better once we turn to net income. On an adjusted basis, Uber Freight lost $13 million in the third quarter compared to a $1 million
profit in the same quarter last year.

Uber said the dismal year-over-year revenue results for its freight business was driven by lower revenue per load and volume. Both are consequences of the
challenging freight market cycle.

Uber Freight isn’t alone. Other more recent entrants to the freight and logistics industry such as Flexport and Convoy have struggled this year. In Convoy’s
case, the business was forced to shut down with its assets gobbled up by Flexport.

Uber Freight continues to plug along, despite these economic headwinds. The question is whether Uber believes in the long-term income potential of freight.

So what?
</item 98>

<item 99>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
WeWork goes bust: As predicted, flexible office-space firm WeWork has filed for Chapter 11 bankruptcy protection, listing over $18.6 billion of debt in a
remarkable collapse for the once high-flying startup co-founded by Adam Neumann and bankrolled by SoftBank, BlackRock and Goldman Sachs.

Slack’s loss, Bumble’s gain: Dating app Bumble announced a doozy this week: It’s replacing founder CEO Whitney Wolfe Herd with Slack CEO Lidiane Jones. Jones
only started as CEO at Slack last year, stepping in for another founder CEO, Stewart Butterfield. Ron and Sarah write that — while Bumble now has a clear line
of succession — the move leaves Slack in a bit of a pickle.

Arrival fails to deliver: Arrival set out eight years ago to make electric vehicle production “radically more efficient.”
</item 99>

<item 100>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Other customers didn’t have the same privileges.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses. In 2021, FTX was facing a “several hundred million dollar” loss from an exploit
in MobileCoin, Wang said. (This seems to have happened in April 2021, though Wang didn’t give a month.) Bankman-Fried directed him to have Alameda take it on
instead, because FTX’s balance sheets were visible to investors and Alameda’s weren’t.

Yesterday, Paradigm Capital’s Matt Huang showed us the balance sheets he received from Bankman-Fried. They didn’t include the loss, which should have been
included in the second quarter numbers he’d received — they showed $63 million in trading expenses and $63 million in “other expenses” in that quarter, and an
estimated profit of $732 million.

“FTX was not fine and assets were not fine.”

By 2022, Alameda’s negative balance on FTX was a cause for concern for Caroline Ellison, Singh, and Wang.
</item 100>
=== END ITEM id=3a1bc38256baf93f ===
