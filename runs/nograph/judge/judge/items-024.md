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

=== ITEM id=eb880624ed1512e8 ===
Gold sentences and facts:
Sentence 0: When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the theft of customer funds to him and a few other employees, Drappi said.
  Fact 0.0: Changpeng Zhao is the CEO of Binance.
  Fact 0.1: Changpeng Zhao announced on Twitter that he intended to acquire FTX.
  Fact 0.2: Ellison confessed the theft of customer funds to Changpeng Zhao and a few other employees when Zhao made that announcement.
  Fact 0.3: Drappi said this.
Sentence 1: Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified.
  Fact 1.0: Ellison was CEO of Alameda Research.
  Fact 1.1: Ellison testified that her tenure as CEO of Alameda Research was an exercise in optics.
Sentence 2: In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from the inherent risk of the crypto markets.
  Fact 2.0: This speaker specifically pointed the finger at Ellison.
  Fact 2.1: Ellison was the former CEO of Alameda.
  Fact 2.2: The speaker said Ellison did not adequately protect Ellison's hedge fund from the inherent risk of the crypto markets.

Retrieved items:
<item 1>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 1>

<item 2>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”
</item 4>

<item 5>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 6>

<item 7>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
She’d testified on direct about Luna, a cryptocurrency token. It had a sister token, Terra, that was a paired algorithmic stablecoin. (If you don’t understand
what that means, it doesn’t matter, because they were both nonsense.) Cohen asked her about Terra/Luna and she pointed out she’d only spoken about Luna, leaving
him to fumble about how to explain the relationship between the two tokens. She kept a straight face on the stand while I chuckled from the press seats.

During the opening statement, Cohen had blamed Ellison for not taking out a hedge on some of the risks Alameda was trading. We heard more about this hedge in
cross-examination, and friends, it was stupid. Forget evaluating the trade itself (hedging being long crypto by selling Nasdaq futures). Was she supposed to
have taken more customer funds to put on the hedge? Was that the defense, that she didn’t take enough of them? Was it that she should have taken them sooner?
What the fuck?
</item 7>

<item 8>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
(I did observe him occasionally shaking his head, and sometimes quivering at points during her testimony, but didn’t have a view of his face.)

At one point, Ellison appeared to even be toying with Cohen

As we were approaching lunch, several jurors looked annoyed, and Cohen looked clueless. He asked Ellison to define what “buy on the way down” meant, as though
it were a term of art. (It means what you think it means, to purchase an asset that’s losing value.) This seemed to puzzle her. At other times, Cohen seemed to
forget what she had testified to, bringing up things she hadn’t said. I don’t know if this was an attempt to trap her in a lie or just poor preparation, but
much like FTX employee Adam Yedidia before her, Ellison was fastidious about making sure a question was clear and her answer was precise.

At one point, Ellison appeared to even be toying with Cohen. She’d testified on direct about Luna, a cryptocurrency token.
</item 8>

<item 9>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Was it that she should have taken them sooner? What the fuck?

When Sassoon got up for a quick redirect, she demolished any points Cohen had attempted to make. But I didn’t really appreciate her cleverness until after
Ellison left the stand, and the jurors left the room. She’d managed to set a neat little trap for Cohen.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval

On the direct examination, near the end, Sassoon asked about an Alameda all-hands meeting, without bringing up many specifics. During the cross, Cohen asked
Ellison what topics were covered in the meeting, while avoiding details. That opened the door for Sasson on redirect to work in that Ellison had confessed to
stealing billions of FTX customers’ money, at Bankman-Fried’s direction.

There had been an open question of whether jurors would hear the tapes of Ellison’s remarks. The testimony set up an argument for the prosecution to bring in
those tapes.
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Then a bigger one. Then an effectively limitless one.

Ellison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have believed you if you told me I would
be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more comfortable with.”

Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When FTX took a loss in
the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s “insurance fund,” meant to
keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.

Even that wasn’t good enough.
</item 10>
=== END ITEM id=eb880624ed1512e8 ===

=== ITEM id=1e495da17e1d3df9 ===
Gold sentences and facts:
Sentence 0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 0.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 0.1: The defense argued that the FTX founder acted in good faith.
  Fact 0.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 0.3: The defense argued that the FTX founder basically got in over his head.
Sentence 1: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 1.0: Judge Lewis Kaplan intervened to ask whether Bankman-Fried had ever been told by Yedidia about that money, in words or in substance.
  Fact 1.1: Bankman-Fried did not admit being told about that money until Judge Lewis Kaplan intervened.
  Fact 1.2: Bankman-Fried admitted he had been told about that money.

Retrieved items:
<item 1>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
regulators and government officials, many of whom he donated to.

Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).

But in early November 2022, that all changed.

FTX’s collapse

Concerns surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8
billion in liabilities as of June 30, 2022.

But there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token
behind FTX.

The balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at
the time.
</item 1>

<item 2>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”
</item 2>

<item 3>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 3>

<item 4>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 4>

<item 5>
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
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 6>

<item 7>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 7>

<item 8>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Wang said he trusted Bankman-Fried’s judgment and didn’t pursue the issue further. Later, though, Alameda’s balance was more negative than FTT and the trading
revenue, and Wang had more conversations with Bankman-Fried about it. Wang said he knew at the time that Alameda was using FTX’s customers’ funds and that he
knew it was wrong.

We heard yesterday about the line of credit Alameda had on FTX, a staggering $65 billion. Today we saw it in the database. The line of credit didn’t start that
big — Wang testified that Bankman-Fried had asked him to increase it a few times because Alameda kept running into its limits. First it was “a few hundred
million,” then a billion, then even that wasn’t enough. It wasn’t clear how Wang arrived at $65 billion specifically, but he said he’d discussed the number with
Bankman-Fried. Other customers didn’t have the same privileges.
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
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 10>
=== END ITEM id=1e495da17e1d3df9 ===

=== ITEM id=b5e20ef2e823eba7 ===
Gold sentences and facts:
Sentence 0: The post came after Swift was seen enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
  Fact 0.0: A post came after Swift was seen enthusiastically cheering on the athlete in the box seats at Arrowhead Stadium.
  Fact 0.1: The sighting of Swift cheering at Arrowhead Stadium fuelled speculation that Swift and the athlete are dating.
Sentence 1: Swifties were immediately on high alert, with some capturing Swift and Kelce leaving the stadium together.
  Fact 1.0: Swifties were immediately on high alert.
  Fact 1.1: Some Swifties captured Swift and Kelce leaving the stadium together.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 1>

<item 2>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Well I know for DAMN SURE that s*** about ‘having kids within the next year’ is NOT coming from Taylor’s side,” one claimed. “She’s got the Eras Tour booked
all through next year, and she’d rather die than cancel a show. But we do know who’s been public about looking for a ‘BREEDER.’”

The comments about Travis and Swift’s romance came days after they seemingly confirmed their relationship. On 22 October, Swift made her return to Arrowhead
Stadium to watch as the Kansas City Chiefs won their game against the Los Angeles Chargers. The occasion marked Swift’s fourth time attending one of Kelce’s
games, amid ongoing reports that they are dating.

Following the event, Chariah Gordon, the partner of Kelce’s teammate Mecole Hardman Jr, took to Instagram to share a snap of the singer and NFL player’s sweet
PDA moment. In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.
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
</item 3>

<item 4>
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
</item 4>

<item 5>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.

In addition to the kiss, the football star also had the perfect reaction to a fan calling him Swift’s boyfriend. After a fan shouted: “Let’s go Taylor’s
boyfriend” while Kelce was waiting for an elevator, he responded with a small smile on his face and by pumping his fist into the air.

Aside from their appearances at sports events, Swift and Kelce have also been photographed together on other occasions, including a date earlier this month in
New York City after their surprise appearances on the season 49 premiere of Saturday Night Live. At the time, sources alleged to Entertainment Tonight that,
during the date, Kelce got a little too close to Swift’s security personnel and allegedly told her bodyguard to step aside.

However, during an episode of his podcast, Kelce explained what he said to the security, and clarified that he didn’t push the bodyguard.
</item 5>

<item 6>
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
</item 6>

<item 7>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 7>

<item 8>
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
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 9>

<item 10>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 10>
=== END ITEM id=b5e20ef2e823eba7 ===

=== ITEM id=4f273f3980272002 ===
Gold sentences and facts:
Sentence 0: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.0: Epic chose to focus on simply painting Google as the bad guy on day one.
  Fact 0.1: Epic's focus on painting Google as the bad guy happened on day one of the trial.
  Fact 0.2: The issue of painting Google as the bad guy will probably come up later in the trial.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: Epic will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 7>

<item 8>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
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
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 10>
=== END ITEM id=4f273f3980272002 ===

=== ITEM id=ef57495f13381ebe ===
Gold sentences and facts:
Sentence 0: For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
  Fact 0.0: The victory has the Swans challenging for a top-eight spot.
  Fact 0.1: The Swans are challenging for a top-eight spot alongside the likes of Gold Coast.
  Fact 0.2: The Swans are challenging for a top-eight spot alongside the likes of Geelong.
  Fact 0.3: The Swans are challenging for a top-eight spot alongside the likes of Collingwood.
  Fact 0.4: The Swans are challenging for a top-eight spot alongside the likes of St Kilda.
  Fact 0.5: The Swans are challenging for a top-eight spot alongside the likes of Carlton.
Sentence 1: With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot.
  Fact 1.0: Sydney is in ninth place.
  Fact 1.1: St Kilda is in eighth place.
  Fact 1.2: Only 0.4 of a percentage point separates Sydney in ninth and St Kilda in eighth.
  Fact 1.3: The win assured the Swans of retaking eighth spot.
Sentence 2: “Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.
  Fact 2.0: The speaker said that her team has put a lot of work into sourcing some wonderful food offerings.
  Fact 2.1: The speaker said that her team has put a lot of work into sourcing entertainment for kids.
  Fact 2.2: The speaker said that her team has put a lot of work into sourcing activities for kids.
  Fact 2.3: The speaker said that her team has had a local focus in promoting its games.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game separating it. It just shows
that this is a really competitive competition.”
</item 2>

<item 3>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.
</item 3>

<item 4>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
WEEK ONE FINALS FIXTURE SATURDAY NOVEMBER 11 First Qualifying Final: Adelaide v Brisbane Lions, 2.35pm local (3.05pm AEDT), Norwood Oval First Elimination
Final: Gold Coast v Sydney, 6.15pm local (7.15pm AEDT), Heritage Bank Stadium SUNDAY NOVEMBER 12 Second Elimination Final: Geelong v Essendon, 1.05pm AEDT,
GMHBA Stadium Second Qualifying Final: Melbourne v North Melbourne, 3.05pm AEDT, Ikon Park The eight was not decided until the final game of the regular season,
when Sydney downed Fremantle by 22 points to leapfrog St Kilda into the final spot, after the Saints had kept themselves in the race with a 20-point win over
Carlton earlier in the day, 7.4 (46) to 3.8 (26).
</item 4>

<item 5>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
“But I go out with on a happy note. I love these girls and I’ll be over at the pancake pocket [Pancake Parlour at Whitten Oval] watching them from now on and
have a few bevs on the sideline.” Roos’ lock in top-four spot; Dogs have issues to work through in off-season North Melbourne secured third spot on the ladder
and the double chance in finals after downing the Western Bulldogs by a whopping 46 points. They will head into finals with momentum but a glaring issue to work
on: kicking accuracy. Despite the comfortable win, they could have done a lot more damage after booting double the number of behinds to goals.

Star midfielder Ash Riddell, who had 36 touches and seven clearances, said she was excited to go up against Melbourne or Adelaide in the first weekend of finals
depending on other results this round, as they were the only two sides who beat them this year.
</item 5>

<item 6>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 6>

<item 7>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 7>

<item 8>
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
</item 8>

<item 9>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 9>

<item 10>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
If the Lions had lost, they were at risk of falling out of the top four with second-season side Essendon chasing their spot. Last season’s best and fairest Ally
Anderson said her side loved playing at its training ground in Springfield, which it has made a fortress, and work well in hot conditions . “It was really hot.
It was a grind… But yeah, it was a good finish,” Anderson told Fox Footy post-game. “Every week that we play here, every time that we play here, we get better
and better. So, it was just a matter of time before we finally got the first win and then it hopefully [will] open the floodgates a bit.”

The Lions next play St Kilda before taking on the reigning premiers Melbourne in the final round. After this weekend, they have staked their claim as being
premiership contenders again. Adelaide will play top-four side North Melbourne next weekend before 18th-placed Western Bulldogs.
</item 10>
=== END ITEM id=ef57495f13381ebe ===
