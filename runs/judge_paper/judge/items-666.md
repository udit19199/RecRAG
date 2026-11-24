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

This file has 5 items.

=== ITEM id=150fbe4c51934d83 ===
Gold sentences and facts:
Sentence 0: He was the richest person in the world under 30 [and] he was going to spend it to save humanity from extinction,” he said.
  Fact 0.0: The speaker was the richest person in the world under 30.
  Fact 0.1: The speaker was going to spend that wealth to save humanity from extinction.
  Fact 0.2: The speaker said this.
Sentence 1: Because FTX had grown so much, he could no longer run both companies, he said.
  Fact 1.0: FTX had grown so much.
  Fact 1.1: Because FTX had grown so much, the speaker could no longer run both companies.
  Fact 1.2: The speaker said this.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 2.2: The defense countered that the FTX founder acted in good faith.
  Fact 2.3: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.4: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 3>

<item 4>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Joe Bankman, Bankman-Fried’s father. He is among the people included in the “small group chat,” the Signal group where the FTX inner circle strategized during
the exchange’s last days.

Barbara Fried’s political action committee, Mind the Gap. It is among the entities that received FTX customer money as political donations. Fried and Bankman
lived together in a Bahamas house purchased for them using FTX investor funds, according to a financial expert called by the prosecution; in a lawsuit filed by
FTX’s bankruptcy estate, Bankman and Fried are alleged to have been given $10 million of FTX customer funds.

Gabe Bankman-Fried’s nonprofit, Guarding Against Pandemics, which got millions from FTX. He was also in a Signal group chat, Donations Processing, that Singh
says was used to coordinate the transfer of FTX and Alameda money to various political causes. In that chat, he can be seen recommending some donations be made
“in the dark” to avoid public relations headaches.
</item 4>

<item 5>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
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
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 10>

<item 11>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 11>

<item 12>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 12>

<item 13>
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
</item 13>

<item 14>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.
</item 14>

<item 15>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.
</item 15>

<item 16>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 16>

<item 17>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 17>

<item 18>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 18>

<item 19>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 19>

<item 20>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 20>
=== END ITEM id=150fbe4c51934d83 ===

=== ITEM id=140a56e43bd1ca4a ===
Gold sentences and facts:
Sentence 0: For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video.
  Fact 0.0: Thursday's game between the Lions and Packers will be aired on Amazon Prime Video.
  Fact 0.1: The broadcast is for NFL fans across the U.S.
Sentence 1: For the masses of NFL fans across the U.S., Thursday's clash between the Bears and Panthers will be aired on Amazon Prime Video.
  Fact 1.0: Thursday's game between the Bears and Panthers will be aired on Amazon Prime Video.
  Fact 1.1: The broadcast is for NFL fans across the U.S.

Retrieved items:
<item 1>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— After an offensive pass interference call backs up the Bears, Chicago shows no interest in taking a shot and hands it off on 3rd & 19. The Panthers will have
solid field position.

10:26 p.m. — The Bears come out with three Foreman runs in four plays. After turnovers doomed them last week, they're playing it conservatively.

10:21 p.m. — Young throws behind the line of scrimmage on 3rd & 15, which doesn't tend to convert. It'll be a Panthers punt, giving the Bears poor field
position in the process.

10:19 p.m. — Now Young hits Adam Thielen for another first down. How about that? Carolina is just about to midfield.

10:18 p.m. — A rare first down for the Panthers as Young throws to Marshall for an 11-yard gain.

Bears 16, Panthers 10

10:13 p.m.
</item 1>

<item 2>
Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-10-19T22:00:04+00:00)
The NFL agreed to a deal with the streaming giant to make it the home of the package for 11 years, so those looking to watch "TNF" nationally will need a
subscription to Amazon Prime to watch the games.

That said, those in the local markets of the teams playing on "Thursday Night Football" can watch the contests without subscriptions to Amazon. CBS affiliate
WJAX (Channel 47) will career the game to those in the Jacksonville area while viewers in the New Orleans market can catch the program on WDSU (Channel 6), an
NBC affiliate.

Amazon's broadcast crew for "Thursday Night Football" will consist of Al Michaels (play-by-play), Kirk Herbstreit (color commentary) and Kaylee Hartung
(sideline reporter). The trio is in their second season working together.

Viewers in Canada will not have to have Amazon Prime to access "Thursday Night Football" in 2023. They will be able to watch the games on DAZN.
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Viewers in Canada can stream every NFL game via DAZN.

MORE: How Joshua Dobbs got his 'Passtronaut' moniker

Bears vs. Vikings start time

Date: Monday, November 27

Monday, November 27 Start time: 8:15 p.m. ET

Monday's game between the Bears and Vikings will kick off at 8:15 p.m. ET (7:15 p.m. local time) from U.S. Bank Stadium in Minneapolis, Minn.

The Bears haven't beaten the Vikings since 2020, with Minnesota 6-1 against Chicago over the last four seasons. The Vikings are 8-3 at home against the Vikings
since 2012.
</item 3>

<item 4>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Johnny Hekker punts it to the 1 and @PatMcAfeeShow nods in approval somewhere#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/WKffggayuG — NFL (@NFL) November 10, 2023

8:28 p.m. — Young is forced to throw it away on third down, so he Panthers will punt from just past midfield.

8:26 p.m. — Young misses a wide open Terrace Marshall Jr. on second down, sailing the pass too high.

8:25 p.m. — Two third downs, two scrambles for a first down by Young. He's not exactly known for his mobility, but he's far from immobile. Now let's see whether
the Panthers can get the passing game going at all.
</item 4>

<item 5>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Thursday's game appeared to be destined to be a showcase of quarterbacking excellence; Jared Goff and Jordan Love have proved themselves to be formidable
options under center since taking the reins of their respective offenses.

But it was the supporting cast that proved so pivotal to Thursday's result for Dan Campbell's side. David Montgomery was a nuisance, slipping through crevices
to puncture the heart of the Packers' defense. When it wasn't him, it was that receiving corp, made up of unheralded youngsters who have all come up trumps in
recent weeks.

As for Love and Co., Thursday's affair was another indication of just how much work there is to be done before they compete for the divisional crown. There were
certainly some flits of delight, throws that highlighted just how much talent Love possesses in his golden arm.

But there were low points, too, misshapen efforts that fell harmlessly to the turf, or harmfully into the opposition's hands.
</item 5>

<item 6>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
the Bears. — Adam Schefter (@AdamSchefter) November 27, 2023

What channel is Bears vs. Vikings on today?

Game: Chicago Bears vs. Minnesota Vikings

Chicago Bears vs. Minnesota Vikings Date: Monday, Nov. 27

Monday, Nov. 27 TV channel: ESPN

ESPN Live stream: ESPN+, Fubo (U.S.) | DAZN (Canada)

Bears vs. Vikings will be aired on ESPN, which has long held the broadcast rights to "Monday Night Football."

Joe Buck will lead the broadcast, serving as the game's play-by-play announcer, with Troy Aikman serving as color analyst. Lisa Salters will provide updates
from the sidelines.

Fans can also stream the game via ESPN+ or Fubo, which offers a free trial. Viewers in Canada can stream every NFL game via DAZN.
</item 6>

<item 7>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Sideline reporter Kaylee Hartung will offer updates throughout the affair.

Fans in the Detroit and Green Bay markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 7>

<item 8>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Bears vs. Panthers start time

Date: Thursday, Nov. 9

Thursday, Nov. 9 Start time: 8:15 p.m. ET

Thursday night's game between the Bears and Panthers is set to kick off at 8:15 p.m. ET (7:15 p.m. local time) from Soldier Field in Chicago.

What channel is Bears vs. Panthers on today?

Game: Chicago Bears vs. Carolina Panthers

Chicago Bears vs. Carolina Panthers Date: Thursday, Nov. 9

Thursday, Nov. 9 TV channel (Chicago) : WFLD (Channel 32)

: WFLD (Channel 32) TV channel (Charlotte) : WSOC (Channel 9)

: WSOC (Channel 9) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Bears and Panthers will be aired on Amazon Prime Video.
</item 8>

<item 9>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Justin Jones brings the heat to force a punt 😤#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/wyEgT3GZ7q — NFL (@NFL) November 10, 2023

9:05 p.m. — A holding call sets the Bears back to 2nd & 20...

9:03 p.m. — After Foreman pushes forward for a key first down, rookie Tyler Scott takes off for 16 yards to take the Bears over midfield. Chicago is looking to
tie the game on this drive.

9:00 p.m. — Sexiest Man Alive Jason Kelce is in the booth with Al Michaels and Kirk Herbstreit to start the second quarter.

Panthers 10, Bears 3

8:55 p.m. FIELD GOAL — Young comes under pressure on 3rd & 11 and just throws it into the ground.
</item 9>

<item 10>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Bears 16, Panthers 10

10:13 p.m. TOUCHDOWN — Foreman runs it in for the score! That's the first offensive touchdown of the night, and it gives the Bears their first lead of the
night.

10:12 p.m. — A well-designed play is broken up by Donte Jackson in the end zone after the Bears pick up the first down.

10:11 p.m. — The Bears are inching their way toward the goal line...it's 3rd & 1 from the five.

10:10 p.m. — Foreman runs 11 yards for a first down, and the Bears are already in field goal range thanks to strong starting field position.

10:06 p.m. — Outside of a 45-yard completion to Mike Strachan, Young is 7-of-16 for 24 yards. It's tough to recall any positive moments for the Panthers'
offense tonight aside from that one play.
</item 10>

<item 11>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
8:09 p.m. — Perhaps the most interesting storyline tonight will be whether Al Michaels can keep a brave face throughout what might not be the most appealing
game.

8:00 p.m. — We're going to keep spirits high tonight despite a matchup of teams with a combined three wins. All football is good football! Let's see how long
this mentality lasts...

7:46 p.m. — It's 52 degrees and clear in Chicago right now, so certainly not as cold as it could be in the Windy City this time of year. Still, it's a rare
colder-weather game for Bryce Young.

7:30 p.m. — The Bears will be donning orange tonight, including an alternate orange helmet.

7:10 p.m. — D.J. Moore is back facing his former team tonight, eight months after he was traded to the Bears in the deal that sent the No. 1 pick to Carolina.

6:50 p.m.
</item 11>

<item 12>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
11:11 p.m.: FIELD GOAL! Carlson's try from distance makes the deficit two possessions.

11:08 p.m.: Not so fast. ... A quick three-and-out gives Green Bay a glimmer. That glimmer grows even brighter after an impressive return by Keisean Nixon.
Let's see what they can do with 3:30 left.

11:04 p.m.: INTERCEPTION! That's wraps. Love misses his target, instead finding the mitts of Jacobs for his second pick of the day. Strong showing by this
Detroit defense. And that should put things to bed.

10:54 p.m.: TOUCHDOWN! Walker's penalty proves vital as Montgomery finds pay dirt from a couple yards out on fourth down. Gutsy call by Campbell. And one that
pays off in spades for Detroit, which extends its lead to 17.

Lions 34, Packers 17

10:50 p.m.
</item 12>

<item 13>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Now let's see whether the Panthers can get the passing game going at all.

Second straight third down that ends in a scramble for a first down by Bryce Young#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/mC5txmYqzP — NFL (@NFL) November 10, 2023

8:23 p.m. — Nice awareness by Bryce Young to run for the first down on 3rd & 7, and the Panthers add five extra yards on an illegal contact penalty against
Chicago.

8:21 p.m. — Donte Jackson comes through with a nice pass breakup on third down, and the Bears will punt shortly after the quick first down.

8:19 p.m. — D.J. Moore instantly makes an impact against his former team with a 16-yard grab.

8:17 p.m. — Bagent and the Bears offense will start with the ball, and we are underway!

8:09 p.m.
</item 13>

<item 14>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
On a night that could be defined by bad offense, the Panthers get a huge early boost from their special teams unit. Carolina leads, 7-0.

Ihmir Smith-Marsette takes one back against his former team 🏠📞#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/4MTENIL0r6 — NFL (@NFL) November 10, 2023

8:35 p.m. — Bagent throws incomplete on third down, and the Bears will punt for the second time tonight.

8:33 p.m. — Darnell Mooney goes for a first down after a nice run by D'Onta Foreman set up a 3rd & 2. The Bears have some breathing room.

8:29 p.m. — Johnny Hekker pins the Bears at the one, so it'll be an uphill battle for Bagent and the offense on this upcoming drive.
</item 14>

<item 15>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The broadcast will be spearheaded by play-by-play announcer Al Michaels and color commentator Kirk Herbstreit. Sideline reporter Kaylee Hartung will offer
sideline updates throughout the game.

Fans in the Chicago and Charlotte markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 15>

<item 16>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Chicago is into field goal range.

Rookie-to-rookie for a key first down 👀#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/nIw42qweEE — NFL (@NFL) November 10, 2023

9:14 p.m. — Bagent hits Roschon Johnson for a first down to give the Bears some life. Chicago started with very strong field position, so this could quickly
become a rare scoring opportunity.

9:11 p.m. — A sack by Justin Jones quickly ends the Panthers' drive. Three-and-out for Carolina.

9:07 p.m. — A promising drive ends in a punt for the Bears. Carolina has a chance to extend its lead to two possessions if Bryce Young can build on that 44-yard
completion he had last time he was on the field.
</item 16>

<item 17>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears will punt, giving the Panthers a chance to tie the game or more.

10:46 p.m. — Bagent finds Kmet for a first down and more. Another nice play by Kmet, who has become a trusted target for Bagent in these four starts for the
rookie.

Bears 16, Panthers 13

10:42 p.m. FIELD GOAL — Pineiro's kick is good, and the Panthers now trail by a field goal.

10:41 p.m. — Young's pass to Marshall is incomplete on third down. Marshall was well-defended, but he probably should've had that. It'll be a field goal try for
Carolina.

10:38 p.m. — You have to give some credit to Carolina's offense for converting, as Young finds Tremble for the first down on 4th & 6. The Panthers turn a 1st &
30 into a fresh set of downs.

10:37 p.m.
</item 17>

<item 18>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
1 pick in March.

MORE: Why Frank Reich's job might already be at risk

The Bears (3-7) are hoping to get Justin Fields back in time for their Week 11 game in Detroit, but they've gotten a shot in the arm from Bagent, who improved
to 2-2 as a starter with the win. The victory on Thursday was a win-win for Chicago, as the franchise owns the first-round pick of the Panthers (1-8).

The Sporting News tracked live scoring updates and highlights from Bears vs. Panthers on "Thursday Night Football." Check out all of the key moments you might
have missed.

Bears vs. Panthers final score

1 2 3 4 F Panthers 7 3 0 3 13 Bears 3 6 7 0 16

Bears vs. Panthers results, highlights from Thursday Night Football

(All times Eastern)

Final: Bears 16, Panthers 13

11:11 p.m.
</item 18>

<item 19>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
10:37 p.m. — A neutral zone infraction pulls the Panthers a bit closer, and they will line up to go for it on 4th & 6...

End of third quarter: Bears 16, Panthers 10

10:32 p.m. — The Panthers run it with Chuba Hubbard on 1st & 30. They're not showing much interest in letting Bryce Young air it out, even while trailing in a
lost season.

10:31 p.m. — It's 1st & 30 for the Panthers after back-to-back penalties. That's one way to erase great field position.

10:29 p.m. — A less than impressive punt by Trenton Gill sets up Carolina well inside Bears territory. If there was ever a time for the Panthers' offense to get
something going...

10:28 p.m.
</item 19>

<item 20>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
: Jordan Love and the Packers heading onto the field ahed of tonight's game! We're about 45 minutes from kickoff at the Pantheon of football excellence, Lambeau
Field.

Lions vs. Packers start time

Date: Thursday, Sept. 28

Thursday, Sept. 28 Start time: 8:15 p.m. ET

Thursday night's game between the Lions and Packers will feature in the primetime slot of 8:15 p.m. ET.

What channel is Lions vs. Packers on today?
</item 20>

<item 21>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
Aaron Rodgers is no longer in Green Bay, and a new love interest is now under center, Jordan Love. It was these Lions that ended the Rodgers era and swept Green
Bay last season. Now they meet again in prime time on one of the league's most hallowed grounds in a battle for NFC North supremacy. The tundra won't be frozen
at Lambeau Field at this point in the season, but the game sure is important, and for a game like this, there's no way we could pass up the chance to win some
cash with our FanDuel single-game DFS lineup.

MORE THURSDAY NIGHT FOOTBALL:

DK lineup | Best props | Betting preview | Start-Sit

Before we break down our picks and strategy, here are the most notable scoring rules for FanDuel contests: four-point passing TDs and half-point PPR.
</item 21>

<item 22>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
1 pick to Carolina.

6:50 p.m. — Standout edge rusher Brian Burns is out with a concussion, dealing a major blow to the Panthers' defense.

Brian Burns, CJ Henderson and DJ Chark among inactives.https://t.co/mL8EYWQKTs — Carolina Panthers (@Panthers) November 9, 2023

6:35 p.m. — Bryce Young threw an interception in only one of his last five games before crashing back to Earth with three picks against the Colts last week.
More than ever, it feels like this game will be won by the quarterback who can avoid turnovers. Bagent has thrown five picks in his last two starts.

6:15 p.m. — Tyson Bagent will get the start for the Bears tonight, so it'll be an undrafted rookie facing a rookie No. 1 pick. Despite that, Chicago is favored.

Bears vs. Panthers start time

Date: Thursday, Nov.
</item 22>

<item 23>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Lions 34, Packers 17

10:50 p.m.: Quay Walker's leaps over a blocker on Riley Patterson's field goal attempt, prompting a unsportsmanlike penalty. That will put Detroit on the cusp
of another score, pending a Packers defensive rally.

10:43 p.m.: Lions beginning to salt the game away with strong running through the tackles. Montgomery and Gibbs creeping through the crevices to push Detroit
down the field. And an impressive toss from Goff into the outstretched hands of Reynolds, as well. The Lions are on the doorstep of the red zone.

10:36 p.m.: TOUCHDOWN! Love scampers into the end zone from short distance to cut that Detroit lead down even more. He got held short of the end zone on the
ensuing two-point effort. But Green Bay is firmly back in it.
</item 23>

<item 24>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
NFL WEEK 4 PICKS: Straight up | Against the spread | Projection model

The Sporting News tracked live updates from Thursday's clash between the Lions and the Packers. Scroll below to check out highlights and live results as two of
the NFC's oldest sides faced off for NFC North glory.

Lions vs. Packers score

1 2 3 4 F Lions 14 13 0 7 34 Packers 3 0 8 9 20

Lions vs. Packers live updates, highlights from Thursday Night Football

(All times Eastern)

End of fourth quarter: Lions 34, Packers 20

11:19 p.m.: Lions complete yet another fourth down, officially putting the game on ice.

11:12 p.m.: Onside kick spills out of bounds, and it's looking like that might be that. Packers have one timeout remaining but Montgomery's fiery run through
the tackles should make that obsolete.

11:11 p.m.: FIELD GOAL!
</item 24>

<item 25>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Carolina has serious questions to answer.

10:05 p.m. — Young gets a few yards back but is sacked by Yannick Ngakoue on third down. The Panthers will punt again.

Yannick Ngakoue gets his team off the field on 3rd down 😤#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/JPt1xc3b93 — NFL (@NFL) November 10, 2023

10:04 p.m. — Bryce Young is now lining up in his own end zone after Miles Sanders was dropped for a six-yard loss.

10:01 p.m. — Punt. Again.

9:57 p.m. — The Panthers get a grand total of one yard (-4 including a delay of game penalty) and punt it away less than a minute into the first half.

End of first half: Panthers 10, Bears 9

9:40 p.m.
</item 25>

<item 26>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
End of first half: Panthers 10, Bears 9

9:40 p.m. FIELD GOAL — In a game that might be lucky to see an offensive touchdown, the Bears deserve some credit for that drive to work their way into field
goal range. They trail 10-9 at the half.

9:39 p.m. — Bagent narrowly avoids a sack and actually runs for positive yardage to move Santos' attempt closer. Chicago is down to three seconds, so here comes
Santos.

9:38 p.m. — A perfectly executed play to get the ball to Cole Kmet, who picks up the first down and then some but still gets out of bounds to stop the clock.
The Bears are in field goal range.
</item 26>

<item 27>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears and Panthers entered Thursday night looking for progress, not playoff contention. If any team achieved that goal, it was Chicago.

It wasn't an aesthetically pleasing game, but that was never the expectation. The Bears grinded out a 16-13 victory behind a terrific defensive performance, a
solid running game, and a few timely throws by Tyson Bagent.

The Panthers jumped out to an early lead on a punt returned for a touchdown by Ihmir Smith-Marsette, but they never found the end zone again. The only offensive
touchdown of the night came in the third quarter, when D'Onta Foreman rumbled his way into the end zone to give Chicago a 16-10 lead.

NFL WEEK 10 PICKS: Straight up | Against the spread

As it turned out, 16 points was all the Bears needed.
</item 27>

<item 28>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
9:28 p.m. — An illegal contact penalty, the second of the night on the Bears, gives Carolina a fresh set of downs.

9:25 p.m. — The Panthers will have 3rd & 3 out of the two-minute warning, just hoping to get in position for some points before halftime. They'll start the
second half with the ball as well.

Panthers 10, Bears 6

9:19 p.m. FIELD GOAL — Bagent is fortunate just to hold onto a bad snap, but his pass is incomplete and brings out Cairo Santos. It's 10-6, Panthers.

9:17 p.m. — A second big play by rookie Tyler Scott, who catches a 15-yard pass from Bagent to give the Bears a fresh set of downs. Chicago is into field goal
range.
</item 28>

<item 29>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears cut the deficit to 7-3.

8:44 p.m. — A false start by Cody Whitehair erases what would've been a good 49-yard field goal...the Bears will keep Cairo Santos out there.

8:43 p.m. — Chicago will try a field goal from 49 yards out after the drive fizzles out.

8:41 p.m. — Foreman has been the Bears' best weapon tonight. He rips off an 11-yard run and now has 43 scrimmage yards in this first quarter.

8:40 p.m. — The Bears rattle off two first downs on their first three plays, with Moore and Foreman grabbing passes from Bagent to take Chicago over midfield.

Panthers 7, Bears 0

8:36 p.m. TOUCHDOWN — Ihmir Smith-Marsette returns the punt 79 yards for a touchdown!
</item 29>

<item 30>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
ET.

What channel is Lions vs. Packers on today?

Game: Detroit Lions at Green Bay Packers

Detroit Lions at Green Bay Packers Date: Thursday, September 28

Thursday, September 28 TV channel (Detroit) : WJBK (Channel 2)

: WJBK (Channel 2) TV channel (Green Bay) : WITI (Channel 6) or WGBA (Channel 26)

: WITI (Channel 6) or WGBA (Channel 26) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video. The broadcast will be
spearheaded by legendary play-by-play announcer Al Michaels. He'll be joined by longtime ESPN college football commentator Kirk Herbstreit, who will offer
occasional musings as an analyst. Sideline reporter Kaylee Hartung will offer updates throughout the affair.
</item 30>
=== END ITEM id=140a56e43bd1ca4a ===

=== ITEM id=b6d314cc74c102de ===
Gold sentences and facts:
Sentence 0: “Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022.
  Fact 0.0: The speaker said, 'Hopefully we can go one better this year.'
  Fact 0.1: The speaker recalled the 2-0 defeat by Canada in 2022.
Sentence 1: After a winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season.
  Fact 1.0: The Swans had a winless campaign in 2022.
  Fact 1.1: The Swans are led by ex-Collingwood player Chloe Molloy.
  Fact 1.2: The Swans have been competitive this season.

Retrieved items:
<item 1>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Instead, in the Davis Cup final in Malaga, Australia will face an Italy team running hot on the heroics of their golden boy Sinner, who beat Djokovic in both
the singles and the doubles in the space of five exhilarating hours in their semi-final on Saturday (Sunday AEDT). Jannik Sinner saved three match points before
going on to beat Novak Djokovic. Credit: Getty Images In one of the great Davis Cup rubbers, Djokovic had held three match points against Sinner and was on the
verge of a completing a relatively workaday 2-0 victory for the Serbs.

But for only the fourth time in his unparalleled career, Djokovic was beaten from match-point up as Sinner roared back to complete his second victory over the
master in just 12 days, winning 6-2, 2-6, 7-5.
</item 1>

<item 2>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
</item 2>

<item 3>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Three looming ODIs against India in the next week and three ensuing T20s will be the first showing of that, with the hosts also a rising force in the white-ball
game.

Advertisement

“Within our group we’re really comfortable with where we’re at, and the progression we’re on,” Healy said.

“Ultimately there’s a big T20 World Cup in Bangladesh towards the end of next year, that is a real carrot for us.

“These are the sort of big trophies that we’re wanting to win.

“And yes, every series in between now and then is important, but at the same time we’re building towards that World Cup.

“Maybe you can ask me that (supremacy) question at the end of the six white-ball games to see how we’re going and we can answer that.

“These next six games will probably give us a better indication about how we’re tracking but also what we might need to improve on.”
</item 3>

<item 4>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The beneficiaries of all this? Lleyton Hewitt’s Australian side, who were happily enjoying a day’s rest at their team hotel further down the Costa del Sol,
watching the drama unfold on TV.

While the Italy tie didn’t finish until 8.15pm after a noon start, Alex de Minaur and co should be fresh for combat in Sunday afternoon’s (Monday AEDT) final as
they seek to lift the Davis Cup for the first time in 20 years. AAP De Minaur, Popyrin power Australia into Davis Cup final Lleyton Hewitt has declared his
close-knit tennis band of brothers will give it “a hell of a shake” to bring back the Davis Cup to Australia after they roared into another final in Malaga.
</item 4>

<item 5>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Alyssa Healy has rejected any suggestion Australia’s stranglehold on international women’s cricket is slipping, saying their Test loss to India is barely a blip
on the team’s record.

Australia stumbled to their first Test defeat to India by eight wickets in Mumbai on Sunday, after being outplayed by the hosts in foreign spinning conditions.

The loss comes after one of the more difficult periods for the Australian side in the past six years, with the retirements of Meg Lanning and Rachael Haynes.

Facebook Twitter Whatsapp Reddit Email Share

After winning their fourth straight global ICC event earlier this year in the 50-over World Cup, Australia retained the Ashes via a drawn multi-format series in
England in July.

That tour included losing four of six white-ball matches, with Australia’s win in the Test enough to have them level on points with England and hold onto the
trophy.
</item 5>

<item 6>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
De Minaur, cheered on by his excited mum, then didn’t need similar heroics to Wednesday’s as he overcame the loss of the first two games.
</item 6>

<item 7>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Following their 2-0 victory over Finland in Friday’s semi-final, executed superbly without a dropped set in singles wins for Alexei Popyrin and Alex de Minaur,
team captain Hewitt said he couldn’t be prouder of their achievement in making back-to-back finals.

And even though they’ll be facing either a Serbian side led by world No.1 Novak Djokovic or an Italian team inspired by young superstar Jannik Sinner in
Sunday’s (Monday AEDT) final, the inspirational de Minaur was adamant: “No one really wants to play us.” He delivered once more for Australia, taking an 11th
victory in his last 13 cup singles matches, sealing Friday’s (Saturday AEDT) win by defeating Emil Ruusuvuori 6-4, 6-3. Alex de Minaur stretches to return the
ball against Finland’s Emil Ruusuvuori.
</item 7>

<item 8>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
The current tour of India is not a multi-format series, but the defeat in the one-off Test marks Australia’s first loss in a four-day game since January 2014.

But quizzed over whether the defeat suggested Australia’s unprecedented period of dominance was ending, Healy was defiant.

“It’s probably not even a blip,” Healy said.

Advertisement

“It (Test cricket) is a format where we’re not overly familiar with, in conditions that we’re not familiar with.

“I see more positives out of this and than any sort of negativity.

“There is always a lot of expectation and external noise about our team and how we’re performing.

“They are so used to seeing us winning, that I think, at times, they forget that we’ve actually seen quite a lot of change over the last 12 to 18 months.

“That kind of goes unnoticed external to our group.”

Australia have spoken about shifting their mindset in white-ball cricket after the losses in England, pursuing a more attacking approach.
</item 8>

<item 9>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 9>

<item 10>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The 22-year-old Sinner, by now an irresistible force, then re-emerged to team up with Lorenzo Sonego and spray massive forehand winners around liberally as they
outplayed a by now weary-looking Djokovic and Miomir Kecmanovic 6-3, 6-4. Djokovic had three match points in the singles rubber. Credit: Getty Images It was a
crushing end to Djokovic’s hopes of ending his triumphant season, which has featured three grand slam victories and an ATP finals showdown victory over Sinner
just six days ago, with a second Davis Cup title. After telling fans to “shut up” earlier in the week, he again got irritated with the crowd in the doubles,
pretending to be the conductor to the Italian fans’ deafening chorus, but though he may be the greatest player who ever picked up a racquet, doubles looks an
alien game to him.
</item 10>

<item 11>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 11>

<item 12>
Tennis News: Kyrgios escalates beef with Becker, change ‘warrior’ Novak had to make, WTA criticised for charisma fail (The Roar | Sports Writers Blog,
2023-12-19T23:54:26+00:00)
The Paris Olympics adds an extra high profile tournament to the four majors next year.

It appears a big leap for Djokovic to win all five – considering he’s yet to claim a Grand Slam of all four – pulling up just short three times.

While there are doubts over the fitness of Rafa Nadal – set to return in Brisbane ahead of the Australian Open – a hungry pack of new talent will be waiting for
a slip up.

Advertisement

“I think that would be a little bit surprising [to win a Golden Slam],” Corretja said.

“Because even if he improves his game so much, he’s getting older. It’s normal that he is getting older and the recovery is not the same.

“I’m sure that he’s going to keep on winning Slams, but it’s going to be tougher and tougher and he knows that.”
</item 12>

<item 13>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Loading Hewitt animatedly embraced Popyrin after his straight-sets victory over 171st-ranked Finn Otto Virtanen, but Australia’s Davis Cup captain would be even
more thrilled if he could repeat the dose against Italy. Australia could also turn to an alternative No.2 singles option, with Jordan Thompson and Purcell
waiting in the wings. “It’s everybody’s dream to play for their country,” Popyrin said. “I looked up to Lleyton, looked up to all the older Aussie legends. To
be able to put on the green and gold, to be able to go out there and play for your country, no matter the format, if it’s this Davis Cup or the old Davis Cup,
you give your all.

“It’s nerves, it’s pressure – you’re playing for everybody back home. It’s something that nothing will ever compare to ... it really was the biggest moment of
my career to get the first live rubber win, to get one foot into the final.
</item 13>

<item 14>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Yet the 22-year-old proved absolutely nerveless, firing down an ace to save the third of the three match points, and then immediately turning the tables by
breaking Djokovic in the very next game amid wild Italian celebrations in the stands at the Palacio de deportes Martin Carpena.

Sinner then served out for a famous win, ending Djokovic’s cherished unbeaten 21-singles winning streak in the Davis Cup and becoming the first man since 2019
to beat the Serb in singles twice in the same year. The drama between the two rivals wasn’t over, as the two teams inevitably both drafted in their top singles
players into the doubles decider. Sinner was again magnificent and still looking ominously fresh after more than four hours on court, with Djokovic’s weak
backhand into the net off the Italian’s second serve in the doubles on match point seeming to sum up his disappointment. The beneficiaries of all this?
</item 14>

<item 15>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 15>

<item 16>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
It was something that I’ll never forget.” Loading Australia, who lost last year’s final to Canada, has enjoyed a soft run on paper to this year’s decider,
eliminating the Czech Republic and Finland – with neither country boasting a top-30 singles player – but will have earned their 29th Davis Cup title if they
account for Italy. Canada defeated Italy in the Davis Cup round-robin matches in September, but bowed out to Finland in the knockout stage without Felix
Auger-Aliassime and Denis Shapovalov, who have both battled injury and form. The final is due to start at 2am AEDT on Monday and will be broadcast on 9GEM.

Sinner ends Djoker’s Davis Cup dream Novak Djokovic’s dream of capping his great season by blasting Serbia past Australia in the Davis Cup final was blown up by
Jannik Sinner’s extraordinary double triumph over the world No.1.
</item 16>

<item 17>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
“Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022. De Minaur had been the hero on Wednesday, pulling out an
extraordinary comeback win over Jiri Lehecka with Australia on the verge of defeat in their quarter-final with the Czech Republic.

But he was happy to pass the laurels to Popyrin, a late squad replacement who’d been asked to take on the troublesome “second singles” role by Hewitt. Loading
“Alexei hasn’t played for Australia in a very long time [two years], had a massive match today against a very high-quality opponent and dug super deep,” said de
Minaur. Popyrin managed to deliver a rock solid game, making only 11 unforced errors to his erratic opponent’s 25, to prevail in just under an hour-and-a-half.
“It was probably the biggest win of my career,” he admitted.
</item 17>

<item 18>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 18>

<item 19>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
“It was probably the biggest win of my career,” he admitted. “To win a match that means so much to us, it’s an honour and something I’ll never forget. Nothing
compares to this.

“I thought I was ready but when I came out and heard the two sets of fans chanting so hard, but it was nerves that I’ve never experienced before.” Loading It
wasn’t easy. The pressure was on with Thanasi Kokkinakis, the injured teammate he’d replaced, Purcell and Jordan Thompson having all lost opening singles ties
in this year’s competition, and Popyrin struggled too against a big-hitting 22-year-old going for broke. But after repelling a set point in the opening stanza,
Popyrin’s solidity in comparison to his increasingly erratic opponent helped silence a noisy contingent of Finnish fans, who’d been hoping to cheer their
underdogs to a first-ever final.
</item 19>

<item 20>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.
</item 20>

<item 21>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
His junior career suggested he was going to be a very good player, and it’s unusual for someone to win the junior French Open and not go on to something big.
Alexei Popyrin celebrates with Australian captain Lleyton Hewitt after his Davis Cup win. Credit: Getty Images “These moments can be a big springboard for
someone like Alexei. He’s a confidence player, and this is a very big moment – whoever he plays, Musetti or Sonego. But he’s got to go out and embrace the
moment, and he’s the type of player who will enjoy it.

“I’d love to give de Minaur a chance with his fighting abilities, but the way Sinner has played the last six months, you’d have to say Sinner is the favourite.”
</item 21>

<item 22>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
His serve was broken in the first set and his court positioning left a big hole for Sinner to blast his forehand through the world No.1 and Kecmanovic, who had
earlier worn down Lorenzo Musetti 6-7 (7-9), 6-2, 6-1 in the tie’s opening singles. The tie, though, will forever be remembered for Sinner’s comeback triumph in
their third brilliant singles duel in less than a fortnight. This time, the Sinner of the Turin group stages re-emerged, the young titan reprising his victory
of 12 days ago when he became the only man since Djokovic’s Wimbledon final loss to Carlos Alcaraz to defeat the 24-time grand slam champion. Here he did
something even more astonishing when, after soaking up immense pressure throughout the deciding set, he stared at 0-40 on his serve at 4-5.
</item 22>

<item 23>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 23>

<item 24>
Tennis News: Kyrgios escalates beef with Becker, change ‘warrior’ Novak had to make, WTA criticised for charisma fail (The Roar | Sports Writers Blog,
2023-12-19T23:54:26+00:00)
I hope that one day we will be closer, to be able to sit and talk and reflect. It would be truly incredible.”

Hewitt at the helm

Australia’s David Cup captain Lleyton Hewitt will return to captain the Aussie team at the United Cup.

Eighteen nations will contest the tournament in Perth and Sydney starting December 29 and ending on January 7.

Advertisement

The tournament began last year with Hewitt and Sam Stosur at the helm.

David Witt, who helped lead the United States to the 2023 trophy, also returns in 2024 for the Americans’ title defence.

Can Djokovic go the distance?

Former Spanish star Alex Corretja has questioned Novak Djokovic durability ahead of a Golden Slam attempt in 2024.

The Paris Olympics adds an extra high profile tournament to the four majors next year.
</item 24>

<item 25>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 25>

<item 26>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.
</item 26>

<item 27>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game separating it. It just shows
that this is a really competitive competition.”
</item 27>

<item 28>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.
</item 28>

<item 29>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 29>

<item 30>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.
</item 30>
=== END ITEM id=b6d314cc74c102de ===

=== ITEM id=f4663553d60874ec ===
Gold sentences and facts:
Sentence 0: “The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the environment,” Goodyear SVP of global operations and CTO Chris Helsel says.
  Fact 0.0: 6PPD-quinone was identified in late 2020
  Fact 0.1: 6PPD-quinone requires more research to understand what impact, if any, it may have on aquatic life or the environment
  Fact 0.2: Chris Helsel is Goodyear's SVP of global operations and CTO
  Fact 0.3: Chris Helsel says the 6PPD-quinone statement
Sentence 1: In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the challenges inherent in sustainability goals.
  Fact 1.0: The report contains a quote from a former VP at an industrial automation vendor
  Fact 1.1: The former VP at an industrial automation vendor said digital solutions provide visibility, analysis, and insight
  Fact 1.2: The digital solutions address the challenges inherent in sustainability goals

Retrieved items:
<item 1>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This will include optimizing the production process of hydrogels as well as sourcing affordable raw materials and minimizing waste generated during the
manufacturing of these materials.

“Ensuring the safety and reliability of hydrogel-based batteries is also necessary,” Fan concluded. “Rigorous testing and evaluation of the potential
environmental impact of hydrogel electrolytes, from raw material extraction to disposal, are vital to meeting sustainability goals and gaining public trust.”

Reference: Peihua Yang, Hong Jin Fan, et al., Hetero-Polyionic Hydrogels Enable Dendrites-Free Aqueous Zn-I2 Batteries with Fast Kinetics, Advanced Materials,
(2023). DOI: 10.1002/adma.202306531

Feature image credit: danilo.alvesd on Unsplash
</item 1>

<item 2>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Truly sustainable tires are becoming a reality, but will it be enough?

Renewable rubber

Bridgestone’s “Wireless-In-Wheel” technology.

If you watched any IndyCar road course race this year, basically any of the half-dozen events where the drivers had to turn right as well as left, you might
have noticed some green flashes on those cars’ tires. It wasn’t just a visual flair to an otherwise visually dull part of the equation.

Those tires contained guayule, part of an agrarian prototype effort from Bridgestone, the supplier of spec tires for that series. It’s a high-profile
demonstration of a new domestically sourced, renewable source of rubber that augments other natural sources like that from the hevea tree and even the Russian
dandelion.

Truly sustainable tires are becoming a reality, but will it be enough?

Bridgestone’s Bill Niaura, executive director of sustainable innovation and circular economy, says guayule is a drop-in replacement with many potential
advantages. Chief among them?
</item 2>

<item 3>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
By the IoT Analytics team.

A new report from IoT Analytics highlights eight notable trends helping to advance and promote digital twins.

Four of these trends are discussed in detail in this article. These trends are shaping the future of the digital twin market and influencing investment
priorities for companies across various industries.

Key insights:

According to our Digital Twin Market Report 2023–2027, the digital twin market is expanding, with a projected CAGR of 30% between 2023 and 2027.

29% of global manufacturing companies have either fully or partially implemented their digital twin strategies. Further, job posts related to digital twins have
increased by 11% compared to October 2021, while openings for other tech topics have declined in the same timeframe.

The report notes eight notable trends helping to advance and promote digital twins, four of which are discussed in this article.
</item 3>

<item 4>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
If not, what investments are required?

2. Data integration: How compatible are our current data sources and systems with digital twin technologies, especially in terms of sensor data (both physical
and virtual/soft)?

3. Interoperability concerns: As we use multiple software solutions across our operations, how important is interoperability between digital twin technologies
and our existing systems?
</item 4>

<item 5>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 4: Initiatives promoting interoperability of digital twins across systems from different vendors

Agreement on integration standards among the various digital twin technology providers is crucial for building cross-system digital twins. With standards,
manufacturers can offer services that can be applied to unique situations with other digital twin technology and software.

Recognizing this, countries and industry organizations have taken several steps to address this need:

Plattform Industrie 4.0 and CESMII

Plattform Industrie 4.0 and the Clean Energy Smart Manufacturing Innovation Institute (CESMII) partnered to establish standards for manufacturers in Germany and
the United States to better enable smart, sustainable competition. The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.
</item 5>

<item 6>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.

DTC digital twin interoperability framework

The Digital Twin Consortium (DTC), a collaborative partnership of industry, academia, and government experts, released the Digital Twin System Interoperability
Framework to “unify a nascent ecosystem of high-value, multi-vendor services that can seamlessly ‘plug into’ a multi-dimensional, interoperable system of
systems.”

DTC and IDTA collaboration

DTC and the Industrial Digital Twin Association (IDTA) announced a liaison agreement to collaborate on standardization requirements, enabling interoperability
through discussions, aligning work in horizontal domains, and collaborating on open-source projects, contributions, and reference implementations.

DTC and OPC Foundation collaboration

DTC and the OPC Foundation announced a liaison agreement to “accelerate the development and adoption of digital twin-enabling technologies,” promoting
interoperability standards and processes to advance the use of digital twins in manufacturing across multiple industries.
</item 6>

<item 7>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This has the effect of mitigating the diffusion back and forth of polysulfide between anode and cathode, the so-called “shuttle effect”, which can cause poor
cycling stability and severe anode corrosion in batteries. The use of a hydrogel electrolyte also has the effect of accelerating the iodine conversion reaction
on the cathode side.

Fan explained that the anode and cathode of the hydrogel have different functional groups that make it both zincophilic (with an affinity for zinc ) and
iodophilic ( with an affinity for iodine), respectively.

On the anode side of the battery, this has the effect of repelling dissolved polyiodides — negatively charged ions composed entirely of iodine atoms — allowing
the zinc to be deposited uniformly, protecting the anode from iodine-caused corrosion.
</item 7>

<item 8>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
This definition captures a macro concept of digital twins, but our report and previous article present a three-dimensional, cuboid model to help classify
digital twins so apple-to-apple comparisons can be made. Each axis of the cuboid represents one dimension of a digital twin:

1. Life cycle phase: The X-axis represents the six life cycle phases a digital twin is used for, from design to decommissioning.

2. Hierarchical levels: The Y-axis represents a digital twin’s five hierarchical levels, from information to multi-system.

3. Use/purpose of implementation: The Z-axis represents the seven most common uses for digital twins, such as simulation and prediction.

Based on this model, there are 210 potential combinations (5 x 6 x 7 = 210); however, our research indicates that many digital twin initiatives are tailored to
multiple combinations.
</item 8>

<item 9>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 3: Partnerships forged in the clouds: Hyperscalers team up with OT and simulation specialists

In the last few years, cloud hyperscalers like AWS and Microsoft Azure have introduced digital twin platforms, e.g., AWS IoT TwinMaker and Azure Digital Twins,
that allow for interconnecting various data sources and building digital twin topology. However, these companies realize they cannot deliver end-to-end digital
twin solutions by themselves and are thus partnering with OT and simulation companies to add capabilities to their networks.

Hyperscalers and OT companies

OT companies provide industrial data management capabilities and connections to millions of local physical assets. By partnering with hyperscalers, OT companies
can reach broader audiences looking for digital transformation solutions by offering typical cloud benefits like storage and computation power.

Example: In 2021, Siemens and AWS announced an expansion of their partnership, with digital twin technology being an area of focus.
</item 9>

<item 10>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
That isn’t bad just for the salmon and the Pacific Northwest’s fishing industry. 6PPD-quinone is in us, too, and it only comes from tires. 6PPD helps tires
resist environmental damage, thus lasting longer between replacements. When 6PPD reacts with ozone, it becomes 6PPD-quinone.

“The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the
environment,” Goodyear SVP of global operations and CTO Chris Helsel says. That’s hardly a mea culpa, but the tire industry genuinely has been an active partner
in studying the problem.

Goodyear and other members of the US Tire Manufacturers Association actually encouraged 6PPD-targeting regulation enacted by the California Environmental
Protection Agency and have provided free tire compound samples to researchers.
</item 10>

<item 11>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
New research aims to improve the stability and safety of alternatives to rechargeable lithium-ion batteries using aqueous zinc and hydrogels.

The demand for rechargeable batteries as a result of the growth of both the personal electronics and electric vehicle markets has grown steadily over the last
decade. Lithium-ion batteries are one of the technologies that have emerged and matured due to this demand.

In 2020, Statista estimated that lithium-ion batteries accounted for 185 gigawatts (GWh) of energy consumed, and this energy demand is projected to reach 2,035
GWh by 2030. Most of this growth is accounted for by the increased desire for electric vehicles as people turn away from fossil fuels.

While it is undeniable that lithium-ion batteries are more environmentally friendly than fossil fuels and while they are also relatively inexpensive, these
batteries aren’t perfect.
</item 11>

<item 12>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Considerations for digital twin vendors

3 questions digital twin vendors should ask themselves based on research findings discussed in this article:

1. Sustainability goals: Given the emphasis on sustainability and the role of digital twins in achieving it, is my company’s digital twin solution well-equipped
to support sustainability objectives? Are there new features we need to develop?

2. Interoperability: Is our digital twin solution currently designed to be interoperable with other systems and technologies, especially those from different
vendors?

3. Product development: How aligned is our product development roadmap with the emerging trends highlighted in the report, especially around end-user and
technological advancements?

Considerations for adopters

3 questions adopters should ask themselves based on research findings discussed in this article:

1. Infrastructure readiness: Do we have the necessary infrastructure and resources in place to support the deployment and effective use of digital twins? If
not, what investments are required?

2.
</item 12>

<item 13>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Digital twin market trends

We will now look at four trends helping digital twin market growth.

Trend 1: Digital twins deployed to meet sustainability goals

Sustainability was discussed in approximately 21% of recent CEO earnings calls and has remained a consistent topic throughout 2023. This pairs with our report’s
analysis that the pursuit of sustainability goals is a tailwind macro factor for the digital twin market.

To achieve their sustainability goals, many companies are exploring digital twins. Due to their ability to simulate real-world conditions and deliver real-time
information, organizations can optimize resource usage, reducing carbon emissions and improving supply and transportation networks.

According to Capgemini Research Institute’s 2022 digital twins report, 57% of organizations agreed that one of the key drivers for their digital twin
investments was improving their sustainability, and 51% agreed that digital twins would help achieve their organization’s environmental sustainability goals.

In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the
challenges inherent in sustainability goals.
</item 13>

<item 14>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
A digital twin strategy as part of an overall digitalization plan can be a crucial capability for asset-intensive industries and needs to encompass the entire
asset lifecycle, process, and value chain from design and operations through maintenance and strategic business planning.”

Example: Aden, a Chinese integrated facility service provider, created a digital twin for one of its commercial centers in Chengdu, China. The digital twin was
designed to help facility managers inspect, maintain, and repair building assets. 3D simulations assist facility managers in visualizing, predicting, and
optimizing energy consumption, and the expected benefits include lower annual energy consumption, water usage, and waste.

Trend 2: Digital twins employed as virtual sensors in complex conditions

Virtual sensors approximate data that otherwise cannot be obtained via physical sensors, often due to physical sensors being impractical, costly, or hazardous
to employ. Companies are now building digital twins to model hardware and calculate data based on other conditions. This not only allows the collection of data
from complex equipment but also enables operators to track equipment performance and predict maintenance and downtimes.
</item 14>

<item 15>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins,” says Nick Molden, CEO of Emissions Analytics, an independent group focused on testing and data about the overall environmental impacts of
transportation.

“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins.”

One such toxin is 6PPD-quinone. A 2020 study from the University of Washington’s Center for Urban Waters found that a dramatic decline in coho salmon
populations was the result of the toxin. The salmon have now been placed on some West Coast endangered lists.

“Shortly after birth, they were essentially going mad and dying,” Molden says. “The researchers finally linked it to this compound in tires because it was
happening after rainfall. The rainfall was washing all the accumulated tire material off the road into the harbors and into the shore.”
</item 15>

<item 16>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
One of the major drawbacks of these lithium-ion batteries is the inherent risk of burning or even explosion as the result of thermal runaway, with overheated
lithium-ion batteries deemed responsible for a number of fires in the past.

A possibility with zinc

One potential substitute for lithium-ion batteries are rechargeable zinc iodine batteries , which have the advantage of being safer, costing less, and being
more friendly to the environment than lithium alternatives.

However, zinc iodine batteries are not perfect. Composed of an negative iodine electrode and a positive zinc electrode , these batteries suffer from issues like
lower energy output compared to commercial non-aqueous lithium batteries and the growth of zinc dendrites on the anodes — projections of material that build on
electrodes and short batteries when they cross from one electrode to another — as well as the corrosion of the zinc anode that result in reduced efficiency and
even short-circuiting.
</item 16>

<item 17>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Based on these observations and additional insights from our report, our current projection estimates that the global digital twin market could grow at a CAGR
of approximately 30% between 2023 and 2027.

Against the backdrop of this growing market, the report notes eight trends that help promote the advancement and adoption of digital twins. Here, we will focus
on four:

1. Digital twins deployed to meet sustainability goals

2. Digital twins employed as virtual sensors in complex conditions

3. Partnerships between cloud hyperscalers and OT and simulation specialists

4. Initiatives promoting interoperability of digital twins across multifaceted systems

Before we dive into these trends, and to provide context, it is helpful to understand how IoT Analytics defines digital twins and the digital twin market.

Defining digital twins and the digital twin market

In our March 2023 Decoding Digital Twins article, we define digital twins as a virtual model replicating the behavior of an existing or a potential real-world
asset, system, or multiple systems.
</item 17>

<item 18>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“For the next five years, if you want to sell tires in California, you need to go through the process of either eliminating 6PPD, or at least properly
researching the alternatives,” Molden says.

“We are collaborating with researchers and other scientists to better understand this transformation product, fill knowledge gaps, and determine next steps,”
says Helsel.

Other initiatives are coming to limit overall tire emissions, including the upcoming Euro 7 regulations, which put caps on tire wear particles as well as other
emissions.

That seems like progress, but Emissions Analytics’ Molden isn’t so sure: “The reason I’m worried about that is I’m sure you can reduce that wear mass, but if
you do that by making the compound more toxic, you could actually make the environmental impact worse.”

“You could actually make the environmental impact worse.”

That could be akin to increasing a car’s fuel efficiency by switching to leaded fuel.

Molden believes that what’s needed is better understanding.
</item 18>

<item 19>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Digital twin market snapshot

Data from our Digital Twin Market Report 2023–2027 indicates that 29% of global manufacturing companies have either fully or partially implemented their digital
twin strategies, marking a noticeable increase from 20% in 2020. Moreover, the proportion of these companies not contemplating the implementation of digital
twins appears to have reduced to 9% in 2023, down from 33.6% in 2020.

Supporting this trend, an analysis of job postings on SimplyHired suggests a growing demand for digital twin expertise. Postings mentioning “digital twin”
increased by 11% in October 2023 compared to just two years prior, while openings for most other tech topics declined in the same timeframe. Notably, among the
60 tech-related skill sets we tracked, digital twin-related skills experienced one of the most significant growths.
</item 19>

<item 20>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Molden believes that what’s needed is better understanding. To that end, he’s recently created the Tyre Emissions Research Consortium, a LinkedIn group with
about 1,000 members. The intent is to break down the silos that individual researchers were working in.

Research into 6PPD replacements is ongoing, and the US Tire Manufacturers Association is also touting research into various ways to mitigate and reduce tire
debris pollution.

Given all that research is happening at an industry level, it’s easy to feel a little helpless about the role we’re all playing in contributing this pollution
to the environment when we’re just trying to get to work. Still, there are things that we can do to help minimize the environmental impact in the interim.

Molden suggests a few things: drive more slowly, drive more smoothly, and choose the right tires. According to Molden, the tires that your car was outfitted
with when new will provide the best wear characteristics.
</item 20>
=== END ITEM id=f4663553d60874ec ===

=== ITEM id=8287fa06672ecc27 ===
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
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The hardcore football fans are pretty much done with the NFL harping about Travis Kelce and Taylor Swift. Reality TV star Rachel Lindsay, a longtime NFL fan,
recently slammed the “obnoxious” coverage, highlighting the “exhausting” uptick in Swifties obsessing over football. But that’s just one example of how fans are
finding this fixation a little too disrespectful towards the sport.

America’s Favorite Video Today

From the league’s humble beginnings in 1920 to the $18 billion juggernaut that it is today, the ride has been rocky with more than just a few small bumps along
the road. But with the ambitious target that NFL Commissioner, Roger Goodell, has set for 2027, is there a new strategy in the 104-year-old league’s master plan
to dominate the world sporting scene?

Why the ‘Traylor’ saga will be the key to Roger Goodell’s $27 billion target?
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
(Top photo: Dustin Satloff / Getty Images)

The Football 100, the definitive ranking of the NFL’s best 100 players of all time, goes on sale this fall. Pre-order it here.
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 4>

<item 5>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.
</item 5>

<item 6>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 6>

<item 7>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.
</item 7>

<item 8>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 8>

<item 9>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 9>

<item 10>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
However, the NFL does not just want to stop at $18 billion. It wants to become a $27 billion league in 4 years. But it must address a few skeletons in its
closet before reaching that goal.

If you were around in the 1970s (that would make you at least 50 years old today), you would know that the NBA did not have the mass appeal that it enjoys now.
With average game attendance hovering around 8,000 a game, it was one of the darkest periods of the basketball league. Drug abuse, brawls on the hardwood court
that sometimes spilled over onto coaches and players – it wasn’t a pretty sight. As reported in the Washington Post, even though there are no reliable figures
on cocaine use by players, “estimates by people in the game range from 40 percent to 75 percent.”
</item 10>

<item 11>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
STEVENSON: Nothing has changed in terms of his efficiency. We can blame the offensive line to some degree but Stevenson has averaged 0.97 yards before contact
per rush this season. That ranks 30th out of 41 qualifying running backs. And if you think that's bad then consider his yards after contact per rush average of
1.79, which is dead last among those same 41 qualifiers.

EXPLOSIVE RUNS: Stevenson has one (1) explosive carry of 12 or more yards. It was against Miami. It went 12 yards. This is a big change from 2022 when he had 21
explosive runs and an explosive run rate of 10%, which was eighth-best.

RECEIVING: Is averaging 3.6 targets per game and catching 67% of them but is getting just 7.3 yards per catch and has one (1) explosive catch of 16 or more
yards.
</item 11>

<item 12>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 12>

<item 13>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 13>

<item 14>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 14>

<item 15>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
JACKSONVILLE JAGUARS | RB Travis Etienne: QB Trevor Lawrence hasn’t quite taken the step forward everyone thought he would, but fortunately for the Jags,
Etienne is emerging as one of the top RBs in the league. He has rushed for multiple touchdowns in three of his past four games and scored a touchdown in five of
the team’s six wins. Etienne ranks third the NFL in rushing touchdowns (seven) and sixth in rushing yards (583).

TENNESSEE TITANS | DT Jeffery Simmons: There haven’t been too many bright spots for the Titans, but Simmons continues to be a wrecking ball on the defensive
line. He leads the team in tackles for loss (six) and ranks second in sacks (4.5). If not for Simmons, Tennessee’s defense would rank a heck of a lot lower than
20th in yards allowed per game (337.8). — Michael Gallagher
</item 15>

<item 16>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
There isn’t a Josh Allen on the Giants or Vikings.

Minnesota in particular finds itself in an interesting spot, with Kirk Cousins approaching free agency and playing well enough, most likely, to keep the Vikings
from climbing too high in the draft order.

Advertisement

“They already gave a contract to (tight end) T.J. Hockenson, and they are going to give one to (receiver) Justin Jefferson,” an exec said. “Now, what? Are you
going into the rookie pool and taking the fourth-best quarterback in the draft? You can look like Atlanta with (Desmond) Ridder pretty quick.”

6. Two-minute drill: There were already many ways to win a game before the Dolphins invented another one.

The Dolphins were one of 223 teams since 2000 to be favored by at least 13.5 points in a regular-season or playoff game. Seventeen of those 223 teams fell
behind by 14 or more. Six of the 17 won anyway.
</item 16>

<item 17>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
But they also haven’t really beaten anyone. They were down 14 points to Carolina. Do you think they are coming back from 13 down against Philly? In my opinion,
no.”

Advertisement

• Dallas Cowboys: Is this team any better than previous Dallas teams that fell short?

What I think: It’s a tough time to evaluate the Cowboys fairly because they were so bad against San Francisco on a big stage in their most recent game. I just
don’t see where Dallas is better or going to get considerably better, especially on offense. And every time they enjoy success, they seem to enjoy it a little
too much. That is a reflection of their owner.

Exec comment: “Run defense and consistency on defense, and then offensively, what is their identity? They just are not very good. Their best games are when they
play great defense and get turnovers, but that is not going to happen enough, especially against the good teams.”

• Detroit Lions: Have they fixed their defense?
</item 17>

<item 18>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
After running the car full-time in 2023, it was announced Thursday that A.J. Allmendinger would go back to full-time Xfinity Series racing for Kaulig in 2024,
though he will continue to run a partial Cup schedule.

The assumption of many had been that the No. 16 would be taken over by Ty Dillon, but that assumption may have been premature. Weaver shared in a recent thread
on X that information suggesting Dillon had been hired to drive the No. 16 was "incorrect," casting ambiguity on both who will join Daniel Hemric at Kaulig next
season as well as the exact plans of Dillon, who has bounced from team to team since Germain Racing shut down following the 2020 season.

Nuts and Bolts
</item 18>

<item 19>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 19>

<item 20>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Cover 7 | Monday A daily NFL destination that provides in-depth analysis of football’s biggest stories. Each Monday, Mike Sando breaks down the six most
impactful takeaways from the week.

The San Francisco 49ers entered Week 6 riding a 15-game winning streak and looking unstoppable with a young quarterback, Brock Purdy, who was making the game
look easy. They exited with a 19-17 defeat at Cleveland, injuries to key players and, for the first time this season, a question mark or two.

Advertisement

With the previously unbeaten Philadelphia Eagles also losing ugly Sunday, this edition of the Pick Six column addresses key questions facing Super Bowl
favorites with one-third of the 18-week regular season nearly complete.

So far this season, we’ve seen the Miami Dolphins set NFL records for offense, only to lose by four touchdowns in their lone game against a true contender.
We’ve seen the Kansas City Chiefs win with defense, sometimes in spite of their offense. We’ve seen the Buffalo Bills incur devastating injuries.
</item 20>
=== END ITEM id=8287fa06672ecc27 ===
