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

=== ITEM id=a4bc87c2cc170410 ===
Gold sentences and facts:
Sentence 0: Hundreds of women have filed lawsuits against Uber claiming the company hasn’t done enough to prevent instances of sexual assault by drivers.
  Fact 0.0: Hundreds of women have filed lawsuits against Uber.
  Fact 0.1: The women's lawsuits claim Uber hasn't done enough to prevent instances of sexual assault by drivers.
Sentence 1: In the third quarter, Uber generated $394 million in operating income and $219 million worth of net income.
  Fact 1.0: In the third quarter, Uber generated $394 million in operating income.
  Fact 1.1: In the third quarter, Uber generated $219 million worth of net income.

Retrieved items:
<item 1>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, ride-hailing, delivery, business model changes,
contra revenue, Uber, startups
</item 1>

<item 2>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Hundreds of women have filed lawsuits against Uber claiming the company hasn’t done enough to prevent instances of sexual assault by drivers. Now a panel of
judges has ruled that more than 80 cases can be consolidated into federal court.

The stakes are high for both parties with implications extending to future Uber riders and drivers. The outcome of the case could result in sweeping changes to
Uber’s platform, which plaintiffs argue will reduce sexual assault and also raise new concerns over privacy.

The upshot? Technology lies at the center of this very human story.

Uber has attempted to address sexual assaults by drivers — which the lawsuits claim Uber has known about since 2014 — through new safety features in its app,
like a 911 button and the ability to share location with a friend. Yet survivors, and their attorneys, say that response has been inadequate, and they’re
calling for better technological solutions like in-vehicle surveillance cameras.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Federal Court, technology, Uber’s platform, Uber’s app, lawsuits against Uber, 911 button, location sharing with a friend, in-vehicle
surveillance cameras, failure to prevent sexual assault, Uber, Uber sexual assault survivors call for in-car cameras, tech upgrades, sexual assault, privacy
</item 2>

<item 3>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Uber sexual assault survivors call for in-car cameras, tech upgrades, Lyft, Charles Breyer, pretrial hearings, sexual assault, class
action lawsuits, Northern District of California, rape
</item 3>

<item 4>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, transport and on-demand sectors, macroeconomic uncertainty, consumer spend, Uber’s Q3 numbers include impressive profitability gains,
slower-than-expected growth, Uber
</item 4>

<item 5>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Checkr, Appriss, Sergio Avedian, The Rideshare Guy, FBI, background checks, video recording feature, joined lawsuit, Uber sexual
assault survivors call for in-car cameras, tech upgrades, fingerprinting, sexual assault
</item 5>

<item 6>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Abrams, Uber sexual assault survivors call for in-car cameras, tech upgrades, Uber in-app safety features, sexual assault
</item 6>

<item 7>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Abrams, Uber sexual assault survivors call for in-car cameras, tech upgrades, Uber safety reports, sexual assault
</item 7>

<item 8>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber Freight, Uber
</item 8>

<item 9>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, FactSet, Refinitiv, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber
</item 9>

<item 10>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Gogoro, California, Rebecca, Los Angeles, Kirsten, Cruise, EV, robotaxis, SecureDrop, Japan Mobility Show 2023, scooter batteries, tiny
car, Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling, UAW
</item 10>
=== END ITEM id=a4bc87c2cc170410 ===

=== ITEM id=b4f0eb117d45e75c ===
Gold sentences and facts:
Sentence 0: Additionally, Caesars has a competitive welcome bonus for new players and runs NBA betting promos for existing players.
  Fact 0.0: Caesars has a competitive welcome bonus for new players.
  Fact 0.1: Caesars runs NBA betting promos for existing players.
Sentence 1: New customers can get up to $1000 in bonus bets if they lose their first bet.
  Fact 1.0: New customers can get up to $1000 in bonus bets.
  Fact 1.1: New customers can get bonus bets if they lose their first bet.
Sentence 2: It has a wide range of bets for the PGA Tour and other major events, such as winning margin, top finishes, and live betting props.
  Fact 2.0: Caesars has a wide range of bets for the PGA Tour.
  Fact 2.1: Caesars has a wide range of bets for other major events.
  Fact 2.2: The bets include winning margin bets.
  Fact 2.3: The bets include top finish bets.
  Fact 2.4: The bets include live betting props.
Sentence 3: We expect Caesars to offer the same offer in VT is has elsewhere, with new sign-ups eligible to receive up to a $1,000 Bonus Bet if their first bet loses.
  Fact 3.0: The writer expects Caesars to offer in VT the same offer it has elsewhere.
  Fact 3.1: New sign-ups are eligible to receive up to a $1,000 Bonus Bet.
  Fact 3.2: New sign-ups are eligible if their first bet loses.

Retrieved items:
<item 1>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Playthrough requirements are attached to site credit and bonus bets. Bonuses can be used to bet on almost any sport. You won't earn back the stake on winning
bonus bets or bets with site credit.

Types of Vermont sportsbook promo and bonus offers

The welcome bonuses and ongoing promotions coming soon from new online betting sites in Vermont will fall into the following categories.

Deposit match bonuses

Get up to 100% of your first deposit matched in bonus funds. Sportsbooks will limit the bonus size you can claim and attach wagering requirements.

Referral bonuses

You share a personal referral code with friends and family, and if they use it to sign up, deposit, and bet, you both get a bonus. Sportsbooks will attach
wagering requirements to the bonus.

First-bet offers

You get a refund if you lose your first bet. Sportsbooks will limit the size of the refund you can claim. Most refunds are bonus bets or site credit, not cash.
That means wagering requirements are attached, and payouts don't include the stake.

Graph facts:
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> NFL

Entities: Vermont, Bonus Bets, wagering requirements, playthrough requirements, site credit, sportsbook promo and bonus offers, deposit match bonuses, referral
bonuses, first-bet offers, Vermont Sportsbook Promos and Sports Betting Launch News, Sporting News, sports betting
</item 1>

<item 2>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
The site offers frequent odds boosts and other bonuses, and new sign-ups also get to join the famed Caesars Rewards program that can earn them more site bonuses
among other benefits.

Here are details explaining how the Caesars Sportsbook welcome offer works:

When registering your new Caesars Sportsbook account, enter the bonus code NEWS1000 .

. Place a first bet within 30 days and if it loses, receive back the equivalent amount in the form of a Bonus Bet worth up to $1,000 .

. After that first bet has been settled, you’ll see the Bonus Bet in your account within two days .

. Use your Bonus Bet within 14 days or else it will expire.

or else it will expire. If you win with your Bonus Bet, you only receive the winnings and not the bet amount back.

BetRivers Vermont promo code

BetRivers Sportsbook is another site that has launched in many states and could be among those coming to Vermont. BetRivers has a lot to offer and many have
found it a great choice elsewhere.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Caesars Rewards
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont

Entities: Caesars Rewards, Vermont, Bonus Bet, NEWS1000, BetRivers Sportsbook, Vermont Sportsbook Promos and Sports Betting Launch News, Sporting News, Caesars
Sportsbook, sports betting
</item 2>

<item 3>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Time limits

These would force you to use a bonus before an expiration date or meet all wagering requirements within a specific timeframe.

Maximum withdrawals

Some offers could limit your ability to withdraw winnings of a particular amount, forcing you to make multiple withdrawal requests on big wins.

Vermont sport-specific betting promos

Vermont has no teams in the four major US pro sports leagues. However, state residents have been known to support Boston-based teams, like the Red Sox (MLB),
the Celtics (NBA), the Bruins (NHL), and the Patriots (NFL).

Because of those teams’ proximity to the state, you can expect Vermont online sportsbooks to run related promos (eg. NFL betting promos) to attract fans.
College sports are big, too, which means several betting promos should also surround local college teams.

How do I claim a Vermont sportsbook bonus?

Many times, all you’ll need to do is sign up for a new account through the links on this page to claim a bonus from a Vermont online sportsbook.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, MLB, Vermont, Vermont Sportsbook Promos and Sports Betting Launch News, Time limits, Maximum withdrawals, Vermont sport-specific betting promos,
Red Sox, Celtics, Bruins, Boston, College sports, NFL betting promos, Vermont sportsbook bonus, NHL
</item 3>

<item 4>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
ESPN Bet brings a very familiar brand to sports fans, but a relatively new one when it comes to online sports betting. After acquiring the old Barstool
Sportsbook earlier in the year, ESPN Bet Sportsbook launched the rebranded site in 17 different states in Nov. 2023. The site has already begun taking advantage
of various tie-ins with the renowned sports programming network which has helped in the effort to establish the site in the overall marketplace.

In other states, the ESPN Bet welcome offer has included a 100% deposit match bonus worth up to $1,000 (with a 20x wagering requirement). ESPN Bet has also run
a welcome promo that invites new sign-ups to place a wager of any amount and receive $200 in Bonus Bets split into four $50 Bonus Bets.

Again, we'll keep an eye out for any news regarding ESPN Bet Sportsbook VT.

Graph facts:
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> NFL

Entities: Vermont, Bonus Bets, ESPN Bet, Barstool Sportsbook, ESPN Bet Sportsbook launch, Vermont Sportsbook Promos and Sports Betting Launch News, Sporting
News, online sports betting
</item 4>

<item 5>
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
</item 5>

<item 6>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
These have included variations on the “bet-and-get” welcome offers described above. Fanatics has featured a bet $5, get $200 in Bonus Bets offer before. They
have also structured the same offer differently by inviting new users to make five bets of at least $10 over five days and receive Bonus Bets each time they do,
again totaling $200.

Fanatics Sportsbook has also taken advantage of the company’s other major business as an official sports apparel and gear provider that partners with all the
major leagues and teams. Drawing on that connection, Fanatics has sometimes made a free jersey, or, rather, money to spend in the Fanatics Store, part of a
welcome offer to new users.

We'll stay tuned to see what happens with Fanatics Sportsbook VT.

ESPN Bet Vermont promo code

ESPN Bet Sportsbook has also applied to operate an online sportsbook in Vermont, and so it could soon be among the new online wagering options for Vermont
bettors as well.

Graph facts:
- Fanatics -PARTICIPATED_IN-> sportsbook launch in Vermont
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Fanatics
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Fanatics Sportsbook
- Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage -MENTIONS-> Fanatics
- Fanatics -MADE-> FanCash

Entities: Fanatics Sportsbook, Fanatics, Vermont, Bonus Bets, Fanatics Store, ESPN Bet, Vermont Sportsbook Promos and Sports Betting Launch News, Sporting News
</item 6>

<item 7>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
11, 2024. BetMGM and ESPN Bet are also among the applicants, making them possible candidates to launch in Vermont in the new year as well. Here are details
regarding those five online sportsbooks and their welcome offers.

FanDuel Vermont promo code

Vermont already knows FanDuel thanks to its daily fantasy sports site. FanDuel Sportsbook is one of the most popular across the US, and FanDuel has already
applied for a license to launch in Vermont. FanDuel has a great welcome offer inviting new sign-ups simply to make an initial $5 wager and get back $200 in
Bonus Bets.

Meanwhile, for Vermont bettors FanDuel is running an Early Sign Up Special that gets you an extra $100 in Bonus Bets for signing up early between now and Jan.
10, 2024. That means you can receive a total of $300 in Bonus Bets to get you started at FanDuel Sportsbook Vermont.

Graph facts:
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> BetMGM
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> BetMGM

Entities: Vermont, ESPN Bet, BetMGM, Vermont Sportsbook Promos and Sports Betting Launch News, US, Sporting News, FanDuel, FanDuel Sportsbook
</item 7>

<item 8>
Graph query row:
entity: Caesars Sportsbook
publisher: Sporting News
article: Line Shopping in Sports Betting - How to Find the Best Odds
connection: NFL
page: Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
passage: Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Line shopping is the key that unlocks the best odds and lines at top sports betting sites.

The process is as simple as checking the lines and odds for your prospective bet at multiple online sports betting sites. The result is quite often better odds
and improved lines.

What is line shopping, and why is it worth your time? Find the answers below in our complete guide to line shopping.

How do sportsbooks set odds and lines?

The best sportsbooks employ oddsmakers and oddsmaking services to set their lines and odds.

Other sports betting apps simply copy those lines. The oddsmakers and oddsmaking services use computer models and algorithms that consider analytics,
statistics, injury and roster news, performance, matchup data, and more to develop the lines and odds.

Betting lines and odds are then subject to change based on the betting public’s reaction and any additional information that comes to light ahead of the game.
Sportsbooks will move the line or change the odds if the public bets heavily on one side.
</item 8>

<item 9>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
BetMGM Vermont promo code

Known as the “King of Sportsbooks,” BetMGM Sportbook is another favorite around the US and will surely be one should it launch in Vermont. BetMGM has applied
for an online sports betting license and should be one of the new sites in VT. When that happens, Vermont bettors should be able to take advantage of BetMGM’s
nice welcome offer of up to $1,500 paid back in Bonus Bets should they lose their first wager on the site.

BetMGM Sportsbook offers lots of ways to bet on many different sports and leagues with a wide range of standard pregame bets (moneyline, spread, totals) plus
lots of props, futures, and live betting opportunities. BetMGM is generous with bonuses and promos, too, with free pools, ongoing refer-a-friend bonuses and
more. There’s also the BetMGM Rewards program for added benefits.

Graph facts:
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> BetMGM

Entities: Vermont, BetMGM, BetMGM Rewards program, Vermont Sportsbook Promos and Sports Betting Launch News, US, Sporting News, BetMGM Sportsbook, online sports
betting
</item 9>

<item 10>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
. Deposit at least $10 into your account.

into your account. Make a first bet, and if you lose get the bet amount back as a Bonus Bet worth up to $500 . (Note that this amount varies from state to state
and could be different in Vermont.)

. (Note that this amount varies from state to state and could be different in Vermont.) You’ll see the Bonus Bet in your account within 24 hours , and then
you’ll have 30 days to use it .

, and then you’ll have . The Bonus Bet has to be used on wagers with odds of -200 or longer, and if you win with it you’ll receive the winnings but not the bet
amount back.

Vermont sportsbook bonuses pros and cons

Vermont Sportsbook Bonus Pros Vermont Sportsbook Bonus Cons Thousands of dollars in value will be up for grabs. Refunds are site credit or bonus bets, not cash.
Bonuses can be claimed easily via links or promo codes. Playthrough requirements are attached to site credit and bonus bets.

Graph facts:
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> NFL

Entities: Vermont, Bonus Bet, sportsbook bonuses, site credit, playthrough requirements, Vermont Sportsbook Promos and Sports Betting Launch News, Sporting
News, sports betting
</item 10>
=== END ITEM id=b4f0eb117d45e75c ===

=== ITEM id=1397ce73eea24915 ===
Gold sentences and facts:
Sentence 0: In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds the performance of other leading gen AI models like OpenAI’s GPT-4.
  Fact 0.0: Google touted Gemini's superior architecture in blog posts and press materials.
  Fact 0.1: Google touted Gemini's superior capabilities in blog posts and press materials.
  Fact 0.2: Google claimed that Gemini meets the performance of other leading gen AI models, such as OpenAI's GPT-4.
  Fact 0.3: Google claimed that Gemini exceeds the performance of other leading gen AI models, such as OpenAI's GPT-4.
Sentence 1: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 1.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 1.1: The case argues that Google siphons off news publishers' content.
  Fact 1.2: The case argues that Google siphons off news publishers' readers.
  Fact 1.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 1.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.

Graph facts:
- One year later, ChatGPT is still alive and kicking -MENTIONS-> GPT-3.5
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> GPT-3.5
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> GPT-3.5
- GPT-4 gets a fine-tuning -MENTIONS-> GPT-3.5
- OpenAI -MADE-> GPT-3.5

Entities: GPT-3.5, TechCrunch, GPT-4, Early impressions of Google’s Gemini aren’t great, Bard, Gemini Pro, Robust Intelligence, AI security researchers, Google,
Gemini, Gemini Ultra
</item 1>

<item 2>
Graph query row:
company: Google
</item 2>

<item 3>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
pic.twitter.com/RKjmkEw2Qr — Filip Piekniewski🌻 🐘:@filippie509@techhub.social (@filippie509) December 6, 2023

And these:

Trying out Gemini Pro: it is pretty disappointing for my example. I asked it to make an analog clock using HTML like this one that ChatGPT made. It can cite
some code from Github but it's off by a few ms… pic.twitter.com/neb42Vzm3m — Mohsen Azimi (@mohsen____) December 7, 2023

GPT 4 still greater than Gemini Pro. Created Tic Tac Toe game with ChatGPT and Bard(Running on Gemini Pro) See video for the result. ChatGPT wrote the code on
first try(First Video). Bard on 3 tries(Second Video).

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> GitHub
- GitHub -SAID-> streamlining effort
- Jason Warner -WORKS_FOR-> GitHub
- GitHub -PARTICIPATED_IN-> GitHub staff reduction announcement
- There’s something going on with AI startups in France -MENTIONS-> GitHub

Entities: GitHub, ChatGPT, TechCrunch, GPT-4, Early impressions of Google’s Gemini aren’t great, Filip Piekniewski, Mohsen Azimi, HTML, analog clock, Tic Tac
Toe game, Bard, Gemini Pro, Google, Gemini
</item 3>

<item 4>
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
</item 4>

<item 5>
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

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, TechCrunch, GPT-4, RAG, 2023 Oscar winners, Early impressions of Google’s Gemini aren’t great, Bard, Gemini Pro, X, Google, OpenAI, Gemini,
generative AI
</item 5>

<item 6>
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
</item 8>

<item 9>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
I tried the same prompt and got a very similar response. ChatGPT, by contrast, gives a bullet-list summary with citations to news articles:

Interestingly, Gemini Pro did provide a summary of updates on the war in Ukraine when I asked it for one. However, the information was over a month out of date:

Google emphasized Gemini’s enhanced coding skills in a briefing earlier this week. Perhaps it’s genuinely improved in some areas — posts on X suggest as much.
But it also appears that Gemini Pro struggles with basic coding functions like this one in Python:

Tried gemini based Bard, and well, it still can't write intersection of two polygons. It's one of those rare relatively simple to express functions that wasn't
ever implemented in python, there is no stack overflow post, and all these models fail on it.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, Stack Overflow, TechCrunch, Early impressions of Google’s Gemini aren’t great, war in Ukraine, Ukraine, Python, Bard, Gemini Pro, X, Google,
Gemini
</item 9>

<item 10>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
Bard on 3 tries(Second Video). pic.twitter.com/cYd9hepcgT — Edison Ade (@buzzedison) December 6, 2023

Just tested Google's Bard with Gemini Pro update. No bugless snake game on 1st try; reported, asked to fix—couldn't. Tried ChatGPT 3.5 free version, got correct
bug-free code on the first attempt! 🚀🐍 #ChatGPT #Bard #Gemini pic.twitter.com/WQfilgG21D — N KIRAN KUMAR (@NKIRANKUMARS1) December 6, 2023

And, as with all generative AI models, Gemini Pro isn’t immune to “jailbreaks” — i.e. prompts that get around the safety filters in place to attempt to prevent
it from discussing controversial topics.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, TechCrunch, Early impressions of Google’s Gemini aren’t great, Bard, Edison Ade, N KIRAN KUMAR, Gemini Pro, Google, Gemini
</item 10>
=== END ITEM id=1397ce73eea24915 ===
