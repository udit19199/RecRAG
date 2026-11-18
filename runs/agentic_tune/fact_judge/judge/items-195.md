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

=== ITEM id=1e792920b8317d1c ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: They get other benefits from Android.
  Fact 0.1: Google could solve this problem today if Google wanted to.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 2>

<item 3>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
As an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one developer that wanted to go the
route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020, worth approximately $360 million,
to keep the developer on Google Play, Epic will argue.

Google, however, will characterize this program as a way to incentivize developers to launch their apps on Android at the same time as they launch on iOS and
plans to dispute Epic’s claims that it prevented developers from opening their own app stores.

Android allows sideloading

Google will make note that, unlike Apple, it also allows apps to be sideloaded on Android devices — something Epic takes advantage of today. It will note this
process is streamlined and helps keep users safe.
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
That sounds fascinating, but if I don’t care about Fortnite or rich companies suing each other, what’s in it for me?

The future of Google’s app store could depend on this trial — both Epic and Google agree on that. Epic wants to break up Google’s alleged monopoly on Android
app stores and payment methods, so developers aren’t stuck paying the “Google Tax” or passing that fee along to you.

But if Epic wins — according to Google, anyhow — it could make Android phones less safe by dismantling basic protections against sideloaded apps, and damage
Android’s ability to compete with the iPhone because it (arguably) can’t run a competitive app store by giving it away for free.

Or maybe you just want juicy dirt on both companies? More on that later!

What exactly is Epic claiming?

Epic argues that Google makes it so difficult for developers and users to get around its Android app store, and that app store’s standard fees, that it’s
created an illegal monopoly that unjustly enriches Google while artificially driving up app prices because other stores can’t compete.
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 6>

<item 7>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”
</item 7>

<item 8>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 8>

<item 9>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=1e792920b8317d1c ===

=== ITEM id=29bf8cd183700c26 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.1: Wang testified that Bankman-Fried said withdrawals were okay as long as the amount was less than FTX's total trading revenue.
Sentence 1: That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.
  Fact 1.0: That answer ultimately lies with the jury.
  Fact 1.1: The jury was selected on Wednesday morning.
  Fact 1.2: The jury was selected before lawyers for the government.
  Fact 1.3: Bankman-Fried then swapped two very different stories.
  Fact 1.4: The two stories are about the former crypto mogul's sudden rise.
  Fact 1.5: The two stories are about the former crypto mogul's almost instantaneous fall.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 2.2: The defense countered that the FTX founder acted in good faith.
  Fact 2.3: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.4: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
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
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 4>

<item 5>
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
</item 5>

<item 6>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 6>

<item 7>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 7>

<item 8>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 8>

<item 9>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 9>

<item 10>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 10>
=== END ITEM id=29bf8cd183700c26 ===

=== ITEM id=1dec5eff65ed3c32 ===
Gold sentences and facts:
Sentence 0: He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not.
  Fact 0.0: He also believes that everything on social media will become perfectly curated for us.
  Fact 0.1: He believes our brains will start to crave things that are not perfectly curated.
Sentence 1: What we have done since then, all the success and failures [matters in our preparation].
  Fact 1.0: What we have done since then, including all the success and failures, matters in our preparation.

Retrieved items:
<item 1>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
And if we can solve it, it will do so much good. It really, really will because we will find out we have more in common than not,” he said.
</item 1>

<item 2>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
I think even swifter than we realize.” The war in the Middle East may even be the tipping point, he theorized, as it’s now “abundantly clear” that we need
something better to figure out the world than the platforms we have today.

The founder, as you may recall, notably resigned from Reddit’s board in 2020 over differences of opinion on moderation and wanting to crack down on communities
about violence and hate. It was a moment where he seemed to be reckoning with what his platform had wrought. At the time, he asked for his board seat to be
replaced by a Black board member. The company agreed, appointing its first Black board member with Y Combinator partner Michael Seibel.

Reddit also later banned several toxic communities, including r/donaldtrup following violence at the U.S. Capitol, and Ohanian says that the site has “gotten
demonstrably better,” since, as well as better as a business.
</item 2>

<item 3>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
If anything, Ohanian credits Twitter (now called X) with coming up with the best system for moderation so far.

“Community Notes, I personally think is a good — it’s the best measure I’ve seen yet. And frankly, I wish we had thought of it first,” he said.

Still, he called out the notion that a platform, like X, can be a town square.

“The public square [analogy]…falls flat. Because if that then implies that it’s a public square, which means like, what? a town or a city? If you’re the person
in charge of it, you’re the mayor. But there’s no democratic election,” Ohanian pointed out. “You’re a CEO. So then you’re like the king? If you want to lean
into the town square thing, fine. Just own it and say, yeah, it’s a town square and I am the infinite dictator, tyrant of it.”
</item 3>

<item 4>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 4>

<item 5>
Graph query row:
article: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand
passage: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“In the game in 2020 there were things you could pinpoint. You can’t say that one game means we can do it as there are a lot of games where we haven’t.

“That moment has to be just a part of it. What we have done since then, all the success and failures [matters in our preparation]. The other team are heavy
favourites, everyone is expecting a New Zealand and South Africa final. We just have to focus on what is required.”

Cheika has maintained that his Los Pumas side was too tight earlier in the tournament but has begun to free themselves up since qualifying for the knockout
stages – and rewarding their fans by doing so.

On inspiring the country?

“It’s not easy,” he said.

“We want to inspire not just by our results but by our ambition, by our ability to get back up and overcome obstacles.
</item 5>

<item 6>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
But as for social media as a whole, of which Reddit would presumably play a part, Ohanian said it brings out the “worst parts of our nature in many ways.”
What’s more, he expressed worries about the desire for people to be first and win on whatever leaderboard a platform features, which then leads to a situation
where misinformation is able to spread.

Humans evolved trusting the faces & voices we knew around the campfire. The idea of believing something bc of experts/institutions you've never met is actually
new to our species. I visited @UVADemocracy + chatted with @karaswisher + @dkroy about what's next for social media. pic.twitter.com/PMBXi7T3vv — Alexis Ohanian
🇦🇲 (@alexisohanian) October 31, 2023

“Now, there’s so much pressure to be first, and UGC — user-generated content — will always win. It won’t always be right, though.
</item 6>

<item 7>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Though he’s glad that access to the technology has broadened, that has come with many complications.

“I don’t like the idea that the internet I’m so nostalgic for was actually an ivory tower,” he said, but on the flip side, he said, we’re all exposed to more
people, in real-time, than we could have ever fathomed. And that can put people at odds with one another and with everyone’s different versions of the truth —
even if their truth is believing in something as far-fetched as the earth being flat, he said, pointing to one example that has a following on Reddit.

Meanwhile, asking platforms like Facebook, Twitter, or Reddit to be in charge of verifying truth and accuracy is a difficult task, he said. Ohanian also called
out TikTok as “an arm for the Chinese government” that’s really good at “feeding us what we want, when we want it” — in other words, not always a fountain of
truth, either.
</item 7>

<item 8>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Does Reddit co-founder Alexis Ohanian have regrets over his creation? The entrepreneur and founder of venture fund Seven Seven Six, speaking at an event at the
University of Virginia, espoused a fairly negative view of the current social media landscape and the creator economy — an environment where he says, everyone
has “been so trained and conditioned now to just get the likes and retweets, get the upvotes…I wish I could be more optimistic about it, but we’re seeing it
play out before our eyes.” Still, he ultimately described himself as a “techno-optimist” — an apparent nod to a16z General Partner Marc Andreessen’s recent
diatribe where he claimed technology could solve all our problems.

“Guilty as charged,” Ohanian said, during his interview with Kara Swisher. “I do think that pendulum will swing back. I think even swifter than we realize.”
</item 8>

<item 9>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Despite the problems with social media and the urgency, in terms of its impact on society, Ohanian believes there’s still some hope. He noted, for example, that
the founders his fund 776 invests in are much smarter and more aware of the issues than he ever was at their age. Perhaps the next platforms will still be
addictive, but in a way that doesn’t “further disrupt democracy,” he mused.

He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not. “I think we’ll
see a rebirth of live gatherings and in-person experiences…the bright side of this is that I hope it then gets us back in touch with our humanity, and the
things that no amount of AI can replace.”

“I want to be an optimist about it,” Ohanian added. “And I think there’s still a path forward but we’re fighting against very, very strong human things.
</item 9>

<item 10>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Welcome to Startups Weekly. Sign up here to get it in your inbox every Friday.

A few months ago, Alexis Ohanian, the co-founder of Reddit, tweeted that if he could go back in time and do one thing differently when he was building Reddit,
he would have spent significantly less time attending events. Personally I have a different experience — I always regret getting on a plane (or in an Uber, for
that matter) to go to an event. However, as I’m traveling back, I’ve never regretted it: There’s always something magical that comes out of going to an event,
in my experience.

Our TC+ team is not interested in anecdotes, however: neither mine nor Alexis’. So, like the data-driven journalists they are, they surveyed more than 50
founders, trying to figure out whether attending events still makes sense in 2023.

Spoiler alert: There isn’t a real consensus, but there are some really good pros and cons.
</item 10>
=== END ITEM id=1dec5eff65ed3c32 ===

=== ITEM id=8f88cf10a88eb28b ===
Gold sentences and facts:
Sentence 0: DeepMind, Google’s premier AI research lab, is expected to debut a next-gen chatbot, Gemini, before the end of the year.
  Fact 0.0: DeepMind is Google's premier AI research lab.
  Fact 0.1: DeepMind is expected to debut a next-gen chatbot, Gemini.
  Fact 0.2: The Gemini debut is expected before the end of the year.
Sentence 1: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 1.0: Google did not release the full model, Gemini Ultra.
  Fact 1.1: Google released only a 'lite' version called Gemini Pro.

Retrieved items:
<item 1>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
(Among other mistruths, Gemini Pro said that Stross contributed to the Linux kernel; he never has.)

Translation doesn’t appear to be Gemini Pro’s strong suit, either. It struggles to give a six-letter word in French:

FYI, Google Gemini is complete trash. pic.twitter.com/EfNzTa5qas — Benjamin Netter (@benjaminnetter) December 6, 2023

When I ran the same prompt through Bard (“Can you give me a 6-letters word in French?”), Gemini Pro responded with a seven-letter word instead of a five-letter
one — which gives some credence to the reports about Gemini’s poor multilingual performance.

What about summarizing news? Surely Gemini Pro, with Google Search and Google News at its disposal, can give a recap of something topical? Not necessarily.

It seems Gemini Pro is loath to comment on potentially controversial news topics, instead telling users to… Google it themselves.

I tried the same prompt and got a very similar response.
</item 1>

<item 2>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Can you save a ChatGPT chat?

Yes. OpenAI allows users to save chats in the ChatGPT interface, stored in the sidebar of the screen. There are no built-in sharing features yet.

Are there alternatives to ChatGPT?

Yes. There are multiple AI-powered chatbot competitors such as Together, Google’s Bard and Anthropic’s Claude, and developers are creating open source
alternatives. But the latter are harder — if not impossible — to run today.

The Google-owned research lab DeepMind claimed that its next LLM, will rival, or even best, OpenAI’s ChatGPT. DeepMind is using techniques from AlphaGo,
DeepMind’s AI system that was the first to defeat a professional human player at the board game Go, to make a ChatGPT-rivaling chatbot called Gemini.

Apple is developing AI tools to challenge OpenAI, Google and others.
</item 2>

<item 3>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
While there is a more…nefarious side to ChatGPT, it’s clear that AI tools are not going away anytime soon. Since its initial launch nearly a year ago, ChatGPT
has hit 100 million weekly active users, and OpenAI is heavily investing in it.

Prior to the leadership chaos, on November 6, OpenAI held its first developer conference: OpenAI DevDay. During the conference, it announced a slew of updates
coming to GPT, including GPT-4 Turbo (super-charged versions of GPT-4, its latest language-writing model), a multimodal API and a GPT store where users can
create and monetize their own custom versions of GPT.

GPT-4, which can write more naturally and fluently than previous models, remains largely exclusive to paying ChatGPT users. But you can access GPT-4 for free
through Microsoft’s Bing Chat in Microsoft Edge, Google Chrome and Safari web browsers.
</item 4>

<item 5>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
I tried the same prompt and got a very similar response. ChatGPT, by contrast, gives a bullet-list summary with citations to news articles:

Interestingly, Gemini Pro did provide a summary of updates on the war in Ukraine when I asked it for one. However, the information was over a month out of date:

Google emphasized Gemini’s enhanced coding skills in a briefing earlier this week. Perhaps it’s genuinely improved in some areas — posts on X suggest as much.
But it also appears that Gemini Pro struggles with basic coding functions like this one in Python:

Tried gemini based Bard, and well, it still can't write intersection of two polygons. It's one of those rare relatively simple to express functions that wasn't
ever implemented in python, there is no stack overflow post, and all these models fail on it.
</item 5>

<item 6>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
Countless other ChatGPT rivals and derivatives have arrived to market since, most recently Amazon Q, a more business-oriented take on ChatGPT. DeepMind,
Google’s premier AI research lab, is expected to debut a next-gen chatbot, Gemini, before the end of the year.

Stella Biderman, an AI researcher at Booz Allen Hamilton and the open research group EleutherAI, told me that she doesn’t see ChatGPT as an AI breakthrough per
se. (OpenAI, which has released dozens of research papers on its models, tellingly never released one on ChatGPT.) But, she says, ChatGPT was a bonafide “user
experience breakthrough” — taking generative AI mainstream.

“The primary impact [ChatGPT] has had [is] encouraging people training AIs to try to mimic it, or encouraging people studying AIs to use it as their central
object of study,” Biderman said.
</item 6>

<item 7>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
what a year it’s been… — Sam Altman (@sama) November 30, 2023

In the months following its launch, ChatGPT gained paid tiers with additional features, including a plan geared toward enterprise customers. OpenAI also
upgraded ChatGPT with web searching, document analyzing and image creating (via DALL-E 3) capabilities. And, leaning on speech recognition, voice synthesis and
text-image understanding models developed in house, OpenAI gave ChatGPT the ability to “hear,” “speak,” “see” and take actions.

Indeed, ChatGPT became priority number one at OpenAI — not simply a one-off product but a development platform to build upon. And, as often happens in a
competition-driven marketplace, it shifted the focus at other AI firms and research labs, too.

Google scrambled to launch a response to ChatGPT, eventually releasing Bard, a more or less comparable AI chatbot, in February.
</item 7>

<item 8>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
pic.twitter.com/RKjmkEw2Qr — Filip Piekniewski🌻 🐘:@filippie509@techhub.social (@filippie509) December 6, 2023

And these:

Trying out Gemini Pro: it is pretty disappointing for my example. I asked it to make an analog clock using HTML like this one that ChatGPT made. It can cite
some code from Github but it's off by a few ms… pic.twitter.com/neb42Vzm3m — Mohsen Azimi (@mohsen____) December 7, 2023

GPT 4 still greater than Gemini Pro. Created Tic Tac Toe game with ChatGPT and Bard(Running on Gemini Pro) See video for the result. ChatGPT wrote the code on
first try(First Video). Bard on 3 tries(Second Video).
</item 8>

<item 9>
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
</item 9>

<item 10>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
Bard on 3 tries(Second Video). pic.twitter.com/cYd9hepcgT — Edison Ade (@buzzedison) December 6, 2023

Just tested Google's Bard with Gemini Pro update. No bugless snake game on 1st try; reported, asked to fix—couldn't. Tried ChatGPT 3.5 free version, got correct
bug-free code on the first attempt! 🚀🐍 #ChatGPT #Bard #Gemini pic.twitter.com/WQfilgG21D — N KIRAN KUMAR (@NKIRANKUMARS1) December 6, 2023

And, as with all generative AI models, Gemini Pro isn’t immune to “jailbreaks” — i.e. prompts that get around the safety filters in place to attempt to prevent
it from discussing controversial topics.
</item 10>
=== END ITEM id=8f88cf10a88eb28b ===

=== ITEM id=6147deece7a17390 ===
Gold sentences and facts:
Sentence 0: “Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022.
  Fact 0.0: The speaker added that the team can hopefully go one better this year.
  Fact 0.1: The speaker recalled the 2-0 defeat by Canada in 2022.
Sentence 1: After a winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season.
  Fact 1.0: The Swans had a winless campaign in 2022.
  Fact 1.1: The Swans are led by Chloe Molloy.
  Fact 1.2: Chloe Molloy is an ex-Collingwood player.
  Fact 1.3: The Swans have been competitive this season.

Retrieved items:
<item 1>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.
</item 1>

<item 2>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 2>

<item 3>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 3>

<item 4>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
It was something that I’ll never forget.” Loading Australia, who lost last year’s final to Canada, has enjoyed a soft run on paper to this year’s decider,
eliminating the Czech Republic and Finland – with neither country boasting a top-30 singles player – but will have earned their 29th Davis Cup title if they
account for Italy. Canada defeated Italy in the Davis Cup round-robin matches in September, but bowed out to Finland in the knockout stage without Felix
Auger-Aliassime and Denis Shapovalov, who have both battled injury and form. The final is due to start at 2am AEDT on Monday and will be broadcast on 9GEM.

Sinner ends Djoker’s Davis Cup dream Novak Djokovic’s dream of capping his great season by blasting Serbia past Australia in the Davis Cup final was blown up by
Jannik Sinner’s extraordinary double triumph over the world No.1.
</item 4>

<item 5>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.
</item 5>

<item 6>
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
</item 6>

<item 7>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
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
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.
</item 9>

<item 10>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 10>
=== END ITEM id=6147deece7a17390 ===

=== ITEM id=50faf4b0cea047ec ===
Gold sentences and facts:
Sentence 0: “It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game in junior — it indicated he was going to be able to do that at some point in the NHL.
  Fact 0.0: Davidson said the player was not playing in the NHL during the last few years.
  Fact 0.1: Davidson said the player's level of performance in junior showed how he could control and dominate a game.
  Fact 0.2: Davidson said the junior performance indicated the player was going to be able to do that at some point in the NHL.
Sentence 1: There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala, among many others.
  Fact 1.0: The player helped guide more than a few future NHL stalwarts at that place.
  Fact 1.1: Mattias Ekholm was one of the future NHL stalwarts the player helped guide.
  Fact 1.2: Ryan Ellis was one of the future NHL stalwarts the player helped guide.
  Fact 1.3: Filip Forsberg was one of the future NHL stalwarts the player helped guide.
  Fact 1.4: Calle Jarnkrok was one of the future NHL stalwarts the player helped guide.
  Fact 1.5: Viktor Arvidsson was one of the future NHL stalwarts the player helped guide.
  Fact 1.6: Kevin Fiala was one of the future NHL stalwarts the player helped guide.
  Fact 1.7: The player helped guide many other future NHL stalwarts besides those six.

Retrieved items:
<item 1>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
If there’s one thing Tim Turk knows, it’s shooting. A self-described “shooting and scoring coach,” Turk has more than two decades of experience working with NHL
stars, national federations and half a dozen NHL teams.

Turk had no idea who Bedard was when he fielded a call from agent Greg Landry of Newport Sports asking him what he was doing tomorrow.

“Tomorrow?” he said. “I’m booked up a year in advance.”

“We have these two players coming in from out west and we want you to see them,” Landry said.

“Listen, I’m booked up,” Turk said again.

“You have to see them,” Landry insisted.

After moving around his schedule, Turk found himself at Gary Roberts’ personal rink in Uxbridge, Ontario.

“Hey guys, I’m Tim Turk,” he said as he stepped into the small locker room.

“My name’s Nate,” said the redhead, extending a hand.

“My name’s Connor,” said the other, standing shorter and extending his.
</item 1>

<item 2>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“It’s a lot of different angles he can give you.”

Sabres first-round pick Zach Benson, who has played on teams with and against Bedard and skates with him in the summers, says it’s Bedard’s ability to set up
the shot with his agility pre-shot that stands out to him.

“He can move left to right like no one I’ve seen before,” Benson said.

“It’s just trying to get out, face him, and hope it hits ya,” said Scott Ratzlaff, Seattle Thunderbirds goalie and Buffalo Sabres prospect, with a smile. “He
can shoot from anywhere and he’s lethal from anywhere. You’ve just always got to be ready just in case he shoots it. And then he’s got a really good toe drag
release, so it’s watching for that and making sure you’re lined up.”

A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting.
</item 2>

<item 3>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.
</item 3>

<item 4>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.
</item 4>

<item 5>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.
</item 5>

<item 6>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
For one, he learned to utilize his teammates more. He accumulated plenty of shots on net early in the season, but they were often from distance and more
individualistic chances. Then, he scored his first NHL goal on a wraparound and his next three goals off passes from teammates.

Advertisement

“Look, I’m sure he’ll figure out the one-on-one and how to score that way or create chances that way,” Blackhawks general manager Kyle Davidson said. “There is
also an element of working off your teammates. You are in the NHL, so everyone is a pretty good player and you’re able to create using more than just yourself.
I think he’s figured that out.”

Bedard found more success on the Blackhawks’ last road trip. He left Chicago with five goals in 11 games and returned from Florida with nine goals in 13 games,
scoring twice against the Tampa Bay Lightning and again against the Panthers.

Bedard wasn’t so reliant on teammates on those goals, either.
</item 6>

<item 7>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
9 pick) went first, and Turk could tell he was a talented and hard-working player with a good shot and real promise.

Then, on Connor’s turn, Turk placed him in a position and watched how he got the puck ready to shoot. Only Connor didn’t stickhandle and prepare for the shot,
he just took the puck and shot it right away.

After watching him run through the same number of shots as Nate did, Turk spoke for the first time.

“Timeout one second, just give me a second here, I’m just trying to see what you’re doing so I’m going to have you take an extra few shots so that I can analyze
where the puck’s starting and where it’s finishing,” he said.

Another round later, he knew that this Connor kid, whose last name he still didn’t know and who was then 16, was very different from even his high-end peers
simply by his proficiency in shooting from a “frontal position.”
</item 7>

<item 8>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The 59-year-old Evason began his coaching career with the WHL’s Calgary Hitmen as an assistant in 1998-99 after finishing out his playing career in Germany,
working his way up the junior hockey ranks over the next handful of seasons. His first NHL job came courtesy of an assistant role with the Capitals immediately
after the 2004-05 lockout, coaching Alex Ovechkin in his first seven NHL seasons.

After departing Washington in 2012, Evason spent the next several seasons working in the Predators organization as the head coach of their AHL affiliate, the
Milwaukee Admirals. There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor
Arvidsson, and Kevin Fiala, among many others. After joining the Wild, he earned a promotion to head coach within two seasons when the Wild fired Bruce Boudreau
late in the 2019-20 campaign.
</item 8>

<item 9>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“So many players pull the puck in but can’t get the puck to release from under their body. His footwork is so underrated in that aspect. If you watch his front
leg, at times it’s literally in the air at release — most have their back leg in the air at release. When he transfers his weight, he actually clears space for
his hands to get tighter because he’s not afraid to actually lunge into the shot.”

A fourth shooting coach, who hasn’t worked with Bedard and also requested anonymity because he works for another NHL club, expressed a little more hesitancy.

Advertisement

“I’m in the minority I’m sure, but I am very curious to see how well his trademark shooting style works at this level,” the coach said. “He really likes to pull
pucks right into his feet.
</item 9>

<item 10>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.
</item 10>
=== END ITEM id=50faf4b0cea047ec ===
