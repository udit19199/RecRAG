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

=== ITEM id=2e17cb6331792d44 ===
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
into your FanDuel account and place a qualifying . Whether your bet wins or loses, you receive $200 in Bonus Bets .

. You can use your Bonus Bets in whatever increments you like, but you have to use your Bonus Bets within seven days or they will expire.

or they will expire. Winning wagers with Bonus Bets award the winnings only and not the bet amounts.

Graph facts:
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> NFL

Entities: Vermont, Bonus Bets, Sportsbook Promos, Vermont Sportsbook Promos and Sports Betting Launch News, Sporting News, FanDuel, sports betting
</item 5>

<item 6>
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
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
This won’t be available on the sportsbook website.

Special event offers

Get a bonus when you place a particular bet on special event odds like World Series odds, Super Bowl betting lines, or March Madness. Special event bonuses are
often bonus bets or site credit. That means wagering requirements are attached, and payouts don't include the stake.

Loyalty offers and programs

Earn points with every bet, and the points are redeemable for rewards like cash and bonus bets.

Important Vermont sportsbook bonus and promo conditions

Terms and conditions will be attached to every bonus and promo offer from Vermont online sports betting sites. Be ready to see these stipulations.

Banking options

Your ability to withdraw the winnings from a bonus could be limited to specific banking methods.

Wagering requirements

You might be forced to bet a bonus amount a minimum number of times before you can withdraw winnings.

Betting limitations

There may be limits on the odds or bet types available for wagering bonus funds.

Graph facts:
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Vermont
- sportsbook launch in Vermont -LOCATED_IN-> Vermont
- Fanatics Sportsbook launch in Vermont -LOCATED_IN-> Vermont
- FanDuel Sportsbook -LOCATED_IN-> Vermont
- Prop Betting - What is a Prop Bet? -MENTIONS-> March Madness

Entities: Vermont, wagering requirements, March Madness, Special event offers, Loyalty offers and programs, Sportsbook bonus and promo conditions, Banking
options, Betting limitations, Vermont Sportsbook Promos and Sports Betting Launch News, Super Bowl, Sporting News, World Series
</item 10>
=== END ITEM id=2e17cb6331792d44 ===

=== ITEM id=603ca1b3132ddb4f ===
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
entity: Google’s hardware event
relationship: MENTIONS
connected_entity: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
article: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
page: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
passage: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 2>

<item 3>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Can you save a ChatGPT chat?

Yes. OpenAI allows users to save chats in the ChatGPT interface, stored in the sidebar of the screen. There are no built-in sharing features yet.

Are there alternatives to ChatGPT?

Yes. There are multiple AI-powered chatbot competitors such as Together, Google’s Bard and Anthropic’s Claude, and developers are creating open source
alternatives. But the latter are harder — if not impossible — to run today.

The Google-owned research lab DeepMind claimed that its next LLM, will rival, or even best, OpenAI’s ChatGPT. DeepMind is using techniques from AlphaGo,
DeepMind’s AI system that was the first to defeat a professional human player at the board game Go, to make a ChatGPT-rivaling chatbot called Gemini.

Apple is developing AI tools to challenge OpenAI, Google and others.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Together, Claude, DeepMind, AlphaGo, Go, AI tools, Apple, TechCrunch,
Bard, AI, Google, Anthropic, OpenAI
</item 8>

<item 9>
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
</item 9>

<item 10>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
(Among other mistruths, Gemini Pro said that Stross contributed to the Linux kernel; he never has.)

Translation doesn’t appear to be Gemini Pro’s strong suit, either. It struggles to give a six-letter word in French:

FYI, Google Gemini is complete trash. pic.twitter.com/EfNzTa5qas — Benjamin Netter (@benjaminnetter) December 6, 2023

When I ran the same prompt through Bard (“Can you give me a 6-letters word in French?”), Gemini Pro responded with a seven-letter word instead of a five-letter
one — which gives some credence to the reports about Gemini’s poor multilingual performance.

What about summarizing news? Surely Gemini Pro, with Google Search and Google News at its disposal, can give a recap of something topical? Not necessarily.

It seems Gemini Pro is loath to comment on potentially controversial news topics, instead telling users to… Google it themselves.

I tried the same prompt and got a very similar response.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, TechCrunch, Early impressions of Google’s Gemini aren’t great, Stross, Linux kernel, Benjamin Netter, Google News, Bard, Gemini Pro,
Google, Gemini, French
</item 10>
=== END ITEM id=603ca1b3132ddb4f ===

=== ITEM id=382b38abc6a27905 ===
Gold sentences and facts:
Sentence 0: In July, speaking on the podcast he shares with his brother, New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City.
  Fact 0.0: In July, Kelce spoke on the podcast New Heights, which he shares with his brother.
  Fact 0.1: Kelce said he intended to give Swift a friendship bracelet with his number on it.
  Fact 0.2: Kelce said he intended to give the friendship bracelet during Swift's Eras Tour concert in Kansas City.
Sentence 1: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 1.0: The game is taking place at Arrowhead Stadium.
  Fact 1.1: Swift has performed at Arrowhead Stadium.
  Fact 1.2: Swift has attended a game at Arrowhead Stadium before.
Sentence 2: Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with pals go on to make headlines.
  Fact 2.0: Swift spoke with Time in the conversation.
  Fact 2.1: Swift is the Anti-Hero singer.
  Fact 2.2: Swift said she tries not to let the paparazzi get to her.
  Fact 2.3: Many of Swift's outings with pals go on to make headlines.
Sentence 3: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 3.0: Swift has shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Kansas City Chiefs game, Arrowhead Stadium, Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game,
friendship bracelet, Taylor Swift, Travis Kelce, TikTok, The Independent, dating, Kansas City Chiefs
</item 1>

<item 2>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Taylor Swift, Travis Kelce, NFL, first game, Taylor Swift reveals secret start to her relationship with Travis Kelce, The
Independent, Kansas City Chiefs
</item 2>

<item 3>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.

Graph facts:
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Time
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: Time, Taylor Swift, Travis Kelce, Kelce’s game on 3 December, public relationship, Taylor Swift reveals secret start to her relationship with Travis
Kelce, The Independent
</item 3>

<item 4>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
The dating speculation first began in July, when Kelce confessed that he tried and failed to give Swift a friendship bracelet with his number on it during her
Eras Tour show. On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24
September. Since then, Swift has attended three more of Kelce’s games.

Kelce showed his support for Swift when he attended her Eras Tour show in Buenos Aires, Argentina, on 11 November. During the concert, Swift gave a shout-out to
the Chiefs player when she changed her “Karma” lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang,
letting out a laugh as fans in the audience screamed in response. The “All Too Well” singer was later seen running into his arms and kissing the NFL player
following the concert, as he whisked her away backstage.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, All Too Well, Anti-Hero, Argentina, friendship bracelet, Travis Kelce says he tried to delete his ‘nonsense’ tweets before they
went viral, Eras Tour show in Buenos Aires, Buenos Aires, dating speculation, Karma, Taylor Swift, Travis Kelce, Chiefs, The Independent
</item 4>

<item 5>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game, Kansas City, Chicago Bears,
Chiefs–Chicago Bears game, post-game celebration, Prime Social Rooftop, 1989 (Taylor’s Version), Taylor Swift, Travis Kelce, Chiefs, Entertainment Tonight, The
Independent
</item 5>

<item 6>
Graph query row:
publisher: The Independent
passage: Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
related_entities: ['Kansas City Chiefs game', 'Arrowhead Stadium', 'Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended
his game', 'friendship bracelet', 'Taylor Swift', 'Travis Kelce', 'Kansas City Chiefs']
</item 6>

<item 7>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Well I know for DAMN SURE that s*** about ‘having kids within the next year’ is NOT coming from Taylor’s side,” one claimed. “She’s got the Eras Tour booked
all through next year, and she’d rather die than cancel a show. But we do know who’s been public about looking for a ‘BREEDER.’”

The comments about Travis and Swift’s romance came days after they seemingly confirmed their relationship. On 22 October, Swift made her return to Arrowhead
Stadium to watch as the Kansas City Chiefs won their game against the Los Angeles Chargers. The occasion marked Swift’s fourth time attending one of Kelce’s
games, amid ongoing reports that they are dating.

Following the event, Chariah Gordon, the partner of Kelce’s teammate Mecole Hardman Jr, took to Instagram to share a snap of the singer and NFL player’s sweet
PDA moment. In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Taylor Swift, Travis Kelce, Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift
romance, Kansas City Chiefs game against the Los Angeles Chargers, Chariah Gordon, Mecole Hardman Jr, Instagram, Los Angeles Chargers, The Independent, Kansas
City Chiefs
</item 7>

<item 8>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Brock Purdy -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> NFL

Entities: NFL, Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game, The Pat McAfee Show, Anti-Hero, episode of
The Pat McAfee Show, Jason Kelce, Taylor Swift, Travis Kelce, The Independent, dating rumours, Donna Kelce
</item 8>

<item 9>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> WSJ Magazine
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: WSJ Magazine, Taylor Swift, Travis Kelce, Taylor Swift reveals secret start to her relationship with Travis Kelce, The Independent
</item 9>

<item 10>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
“OF ALL 230 SONGS IN THE DISCOGRAPHY I’M SICK,” one fan commented, while another person wrote: “It’s unbelievable how invested I am in this.”

“I’m gonna need to go back and see all photos of him from the last three months and dissect if he’s wearing friendship bracelets,” a third wrote.

In other photos from the charity event - which were shared to Instagram by Jordan’s KC photography - Kelce wasn’t seen wearing the friendship bracelet, leading
many people to believe that he received the bracelet from a fan that day.

Two days after his charity event, the “All Too Well” singer was spotted cheering for the Kansas City Chiefs tight end from the box seats at Arrowhead Stadium in
Kansas City, Missouri. Not only was she seen sitting next to Kelce’s mother, Donna, but she also appeared to yell “let’s f***ing go” in response to Kelce’s
third-quarter touchdown reception.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Arrowhead Stadium
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Arrowhead Stadium
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Arrowhead Stadium
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Arrowhead Stadium

Entities: Arrowhead Stadium, Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game, charity event, Jordan’s KC
photography, All Too Well, Kansas City, Missouri, friendship bracelet, Taylor Swift, Travis Kelce, Donna, game, Instagram, The Independent, Kansas City Chiefs
</item 10>
=== END ITEM id=382b38abc6a27905 ===
