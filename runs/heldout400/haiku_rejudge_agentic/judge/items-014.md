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

=== ITEM id=02aade55544c2ac1 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that Wang heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.1: The withdrawals were okay as long as the amount was less than FTX's total trading revenue.
Sentence 1: That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.
  Fact 1.0: The answer to the question ultimately lies with the jury.
  Fact 1.1: The jury was selected on Wednesday morning.
  Fact 1.2: The jury was selected before lawyers for the government.
  Fact 1.3: Bankman-Fried swapped two very different stories of the former crypto mogul's sudden rise.
  Fact 1.4: Bankman-Fried swapped two very different stories of the former crypto mogul's almost instantaneous fall.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power, and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
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

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Mark Cohen
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Adam Yedidia
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission

Entities: The jury finally hears from Sam Bankman-Fried, Alameda Research, jury hears from Sam Bankman-Fried, Singh, FTX, Sam Bankman-Fried, The Verge
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Gary Wang

Entities: Gary Wang, Securities and Exchange Commission, misappropriation of customer funds, Alameda, Christian Everdell, spot margin program, Sam Bankman-Fried
was a terrible boyfriend, FTX, Sam Bankman-Fried, trial, fraud, The Verge
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

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda Research, FTX, Nishad Singh, Wang, Sam Bankman-Fried, The Verge
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.

Graph facts:
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> crypto
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Gary Wang
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> The New York Times
- The FTX trial is bigger than Sam Bankman-Fried -MENTIONS-> Sequoia Capital

Entities: Gemini, fraud trial, The FTX trial is bigger than Sam Bankman-Fried, FTX, Sam Bankman-Fried, Kraken, Coinbase, crypto industry, The Verge
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, risk management, consumer protection, conflicts of interest, superseding indictment, The FTX trial is bigger than Sam Bankman-Fried, FTX, Sam
Bankman-Fried, government, The Verge
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, FTX trial, The FTX trial is bigger than Sam Bankman-Fried, extensive media tour, interviews, Andrew Ross Sorkin, Good Morning
America, The New Yorker, FTX, Sam Bankman-Fried, The Verge
</item 6>

<item 7>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
“I worried if I didn’t, I would lose touch with the actual concerns of the customers,” he testified. What he did not do was create a risk team, which he is now
characterizing as a “big mistake.”

Risk is an inherent part of a futures exchange, which is even more like a casino than regular cryptocurrency. Not having a risk team, when you are any kind of
financial anything, is certainly a choice. It is especially a choice when you go around telling everyone your crypto exchange is very good and safe.

FTX’s big selling point was its “risk engine,” which was supposed to prevent big losses that would then be spread around all the rest of the customers. But
Bankman-Fried testified that in 2020, the “risk engine was effectively sagging under the weight” of the exchange’s rapid growth. So its time to liquidation went
up — it took minutes to determine which accounts needed to be liquidated.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Mark Cohen
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Adam Yedidia
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission

Entities: The jury finally hears from Sam Bankman-Fried, risk team, risk engine, testimony, cryptocurrency, FTX, Sam Bankman-Fried, risk, The Verge
</item 7>

<item 8>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda Research

Entities: Bankman-Fried, Alameda, FTX, Wang, FTT, FTX jury, Today the FTX jury suffered through a code review, The Verge
</item 8>

<item 9>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda, Sam Trabucco, MobileCoin loss, Singh, FTX, Wang, Sam Bankman-Fried,
MobileCoin, The Verge
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It was turned on in July. No other customer trading on FTX had “allow_negative” privileges — only Alameda. Wang’s testimony suggests that the fraud at FTX
started very early since the same day that allow_negative was switched on, Bankman-Fried tweeted that Alameda’s account “was like everyone else’s.”

The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating. For instance, Alameda Research first had a
negative balance in late 2019, Wang testified. Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money
from the account as long as the amount was less than FTX’s total trading revenue. A little later, Wang discovered that Alameda was negative by $50 million more
than FTX’s revenue. Bankman-Fried told him it was fine, and it just didn’t account for the FTT token that the two of them had launched. Wang trusted him.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda

Entities: Bankman-Fried, Alameda Research, In the end, the FTX trial was about the friends screwed along the way, allow_negative, FTX trial, FTX, Wang, FTT
token, fraud, The Verge
</item 10>
=== END ITEM id=02aade55544c2ac1 ===

=== ITEM id=e5b2b1488bbe9f8d ===
Gold sentences and facts:
Sentence 0: He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not.
  Fact 0.0: He also believes that after everything on social media becomes perfectly curated for us, our brains will start to crave things that are not.
  Fact 0.1: He believes that social media becomes perfectly curated for us.
Sentence 1: What we have done since then, all the success and failures [matters in our preparation].
  Fact 1.0: What we have done since then, including all the success and failures, matters in our preparation.

Retrieved items:
<item 1>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
But as for social media as a whole, of which Reddit would presumably play a part, Ohanian said it brings out the “worst parts of our nature in many ways.”
What’s more, he expressed worries about the desire for people to be first and win on whatever leaderboard a platform features, which then leads to a situation
where misinformation is able to spread.

Humans evolved trusting the faces & voices we knew around the campfire. The idea of believing something bc of experts/institutions you've never met is actually
new to our species. I visited @UVADemocracy + chatted with @karaswisher + @dkroy about what's next for social media. pic.twitter.com/PMBXi7T3vv — Alexis Ohanian
🇦🇲 (@alexisohanian) October 31, 2023

“Now, there’s so much pressure to be first, and UGC — user-generated content — will always win. It won’t always be right, though.

Graph facts:
- Bride defended after asking mother to walk her down the aisle instead of stepfather -MENTIONS-> Reddit
- The people who ruined the internet -MENTIONS-> Reddit
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Reddit
- How to get Pumpkin Tickets, Jack-o’-Head armor in Monster Hunter Now’s Halloween event -MENTIONS-> Reddit
- 2023 in social media: the case for the fediverse -MENTIONS-> Reddit

Entities: Reddit, TechCrunch, misinformation, Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society, Kara Swisher,
user-generated content, @UVADemocracy, @dkroy, Alexis Ohanian, social media
</item 1>

<item 2>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
PARIS – The All Blacks have downplayed Rieko Ioane’s gestures towards Ireland fans and retiring great Johnny Sexton, as the three-time world champions said they
have learned to “respect Argentina” ahead of their semi-final on Saturday (6am AEDT).

It comes as Michael Cheika embraced the underdog status and said he was hoping Los Pumas “inspired” Argentina “not by our results but by our ambition”, as the
former Wallabies coach readies for another clash against the All Blacks and a second semi-final.

Facebook Twitter Whatsapp Reddit Email Share

Ioane copped a spray from Sexton following their 28-24 nail-biting win over Ireland.

It came after the lethal back cupped his ear at Irish supporters at the Stade de France after the final whistle, as well as putting his finger to his lips in a
‘shush gesture’.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues -MENTIONS-> Johnny Sexton
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea

Entities: Johnny Sexton, Michael Cheika, semi-final, 28-24 win over Ireland, ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on
‘favourites’ New Zealand, Paris, Stade de France, Wallabies, All Blacks, Argentina, Ireland, Rieko Ioane, The Roar
</item 2>

<item 3>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
I’ve got the experience of being in difficult situations before,” Jones said.

“This is not an unusual situation. You’ve got a team that was struggling for a period of time, you’ve got underneath it a system that’s not supporting it and at
some stage you bottom out.

“This is the opportunity now to change the team- as we’ve started with picking younger players with bright futures – and then we’ve got to look at the
development system underneath as well.

“It’s not a simple equation that the Wallabies are bad. Let’s be serious about this. The Wallabies are not where they need to be but underneath that we need to
fix the system to go forward. And if we can do that now we’ve got a rosy future.”

Sports opinion delivered daily
</item 3>

<item 4>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“To inspire is not about results, it’s not just one player who played well; no, it’s about our behaviour in general, our ability to be ready to come back from
difficult times and seizing our opportunities.

“There are not a lot of opportunities in life – us playing in a semi-final is even rarer and we want to have no regrets.”

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, All Blacks, The Roar
</item 4>

<item 5>
Graph query row:
article: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand
publisher: The Roar
person: Michael Cheika
connection: MENTIONS
connected_entity: Sam Cane
passage: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
PARIS – The All Blacks have downplayed Rieko Ioane’s gestures towards Ireland fans and retiring great Johnny Sexton, as the three-time world champions said they
have learned to “respect Argentina” ahead of their semi-final on Saturday (6am AEDT).

It comes as Michael Cheika embraced the underdog status and said he was hoping Los Pumas “inspired” Argentina “not by our results but by our ambition”, as the
former Wallabies coach readies for another clash against the All Blacks and a second semi-final.

Facebook Twitter Whatsapp Reddit Email Share

Ioane copped a spray from Sexton following their 28-24 nail-biting win over Ireland.

It came after the lethal back cupped his ear at Irish supporters at the Stade de France after the final whistle, as well as putting his finger to his lips in a
‘shush gesture’.
</item 5>

<item 6>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Ian Foster, who made two changes to his starting side and three in total, was forced to defend his centre’s actions while twisting the knife into Sexton’s
finish.

“It happens on most sporting parks when tensions are high and a lot at stake,” Foster said.

“Unfortunately recently it’s also included players saying a few things to referees. Is it right for the game? I don’t know. You don’t hear a lot of players
complaining about it.”

Advertisement

Captain Sam Cane added: “It’s players’ different personalities and emotions running high in those moments. There’s always people who have different tactics but
it’s always just left out there.”

Foster recalled Sam Whitelock to the starting side in the only change to the pack.

Whitelock, whose breakdown penalty win after 37 phases denied Ireland a match-winning penalty, has been recalled and will partner Scott Barrett in the
second-row. Brodie Retallick will come off the bench.

Graph facts:
- Sam Cane -ACCUSED_OF-> high shot
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Sam Whitelock
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Sam Cane
- Sam Cane -PARTICIPATED_IN-> RWC final
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Sam Whitelock

Entities: Sam Cane, Sam Whitelock, Brodie Retallick, Scott Barrett, Sexton, Players saying things to referees, Players’ personalities and emotions running high,
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, All Blacks, Ireland, Ian Foster, New
Zealand, The Roar
</item 6>

<item 7>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
“Firstly, you’ve never heard us say we’re favourites,” he said.

‘We know these games are do or die. We’re going to have to improve our performance. The stage gets bigger at this time in the tournament and you’ve got to grow
your game. That’s our strong focus. The best team on the night wins – Argentina has done that to us.”

Cheika however was happy to remind the All Blacks who were favourites.

“”New Zealand has always been an example in rugby, a benchmark. They make you think of high-level skills in a really open game but there are always threats in
the lineout, mauls and rucks,” said Cheika, whose Wallabies lost in the 2015 final.

“You have to take the moments you are in the game and see them coming at you at different times.

Advertisement

“In the game in 2020 there were things you could pinpoint.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, Argentina, 2015 final, game in
2020, Wallabies, rugby, All Blacks, New Zealand, The Roar
</item 7>

<item 8>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Wallabies
- Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies -MENTIONS-> Wallabies
- The story behind anguished Wallabies star's photo -MENTIONS-> Wallabies
- RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach -MENTIONS-> Wallabies
- Samu Kerevi -SAID-> Wallabies

Entities: Wallabies, Australia, AFL, RA, rugby league, Australian Super Rugby sides, QRU, ‘Right people in right roles’: QRU breaks silence on RA’s reform
hopes, backs review into Wallabies’ RWC flop, ACT Brumbies, The Roar
</item 8>

<item 9>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
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

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, Los Pumas, South Africa, All
Blacks, New Zealand, The Roar
</item 9>

<item 10>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
NEW PODCAST! Michael Atkinson and Sam Worthington preview a thrilling final round of pool games to decide the quarter-finalists at the Rugby World Cup in
France﻿

He wanted to regain form at club level ﻿with Urayasu D-Rocks but his hand remained up for Test selection.

"I chose to leave Australian shores and understand the ruling around that and what it holds," he said.

"I've got to perform for my club first and if you get the call up, you get the call up. If you don't, I'll take my three months break and enjoy that."

Kerevi is an unabashed Jones fan.

Eddie Jones with fans during a Wallabies training session at Stade Roger Baudras. (Getty)

"He's up at 4am working hard. We catch him in the gym, he's working hard and I swear he doesn't even sleep.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, Wide World Of Sports, Wallabies, The story behind anguished Wallabies star's photo, Stade Roger Baudras, Michael Atkinson, Sam Worthington,
Kerevi, Urayasu D-Rocks, Test selection, Rugby World Cup, Eddie Jones, Getty
</item 10>
=== END ITEM id=e5b2b1488bbe9f8d ===

=== ITEM id=bfb1c4743371b23a ===
Gold sentences and facts:
Sentence 0: Postponements of such reports could complicate things for the Fed, which has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.
  Fact 0.0: Postponements of such reports could complicate things for the Fed.
  Fact 0.1: The Fed has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.
Sentence 1: With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest rates and could soon turn to cutting them.
  Fact 1.0: Inflation is down from its peak two summers ago.
  Fact 1.1: Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest rates.
  Fact 1.2: Wall Street is hopeful that the Federal Reserve could soon turn to cutting interest rates.

Retrieved items:
<item 1>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.

Graph facts:
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Evolution
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Brent crude
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> benchmark US crude
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> Brent crude
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> gold miners

Entities: benchmark US crude, Brent crude, ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields, Thursday’s inflation report,
Seema Shah, Principal Asset Management, Fed’s next move on interest rates, interest rate hike, Fed, The Age, the economy, bond yields, 10-year Treasury yield,
crude oil prices, ASX
</item 1>

<item 2>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Fed put such pressure on the economy intentionally, hoping to starve inflation of its fuel. It wants the job market to cool, particularly pay raises going
to workers. The Fed fears too-strong pay gains could create a vicious cycle that keeps inflation high.

Analysts said Friday’s jobs report offered encouraging signals for the Fed, with average hourly earnings rising less in October from September than expected,
though it doesn’t mean the job is done. Treasury yields in the bond market tumbled immediately after the jobs report, releasing more of the pressure that had
built up on Wall Street. The yield on the 10-year Treasury eased to 4.52 per cent from 4.67 per cent late Thursday and from more than five per cent last week,
when it hit its highest level since 2007. In the US, traders are moving up expectations for when the first cut to interest rates by the Fed could happen,
potentially by the summer, according to data from CME Group. Such cuts can act like steroids for financial markets.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> financial markets
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> financial markets
- There are no sacred numbers -MENTIONS-> financial markets
- Psagot worth 80% less since Rani Zim bought it -MENTIONS-> financial markets
- Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions -MENTIONS-> financial
markets

Entities: financial markets, The Sydney Morning Herald, US, economy, bond market, interest rates, ASX to rise after interest rate hopes boost US shares, ASX, US
shares, Wall Street, Treasury yields, The Fed, job market, pay raises, Friday’s jobs report
</item 2>

<item 3>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“The bond market is retracing,” he said. “There’s also been strength in resource stocks amid rumours that the Chinese government might launch a substantial
stimulus package.” Clark said interest-rate sensitive sectors climbed on increased probabilities of an easing in interest rates. “Consumer discretionary and
information technology companies are winners in lower-interest rate environments,” he said. As more annual general meetings kick off in the next few weeks,
Clark said he would be looking at trading updates, especially of companies that didn’t give solid guidance in their last update.

Elsewhere, Wall Street advanced overnight, with the S&P 500 and Nasdaq touching three-week highs as dovish comments from the Fed officials pushed Treasury
yields lower. Following remarks from other top Fed officials on Monday, Atlanta Fed President Raphael Bostic said the US central bank did not need to raise
interest rates any further, and that he saw no recession ahead. Oil edged lower, with West Texas Intermediate settling below $US86 a barrel following its
biggest rally since April.

Graph facts:
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Fed
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> Fed
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> Fed
- Powell Says Fed to Be Careful, Won’t Hesitate to Hike If Needed -MENTIONS-> Fed
- ASX to rise after interest rate hopes boost US shares -MENTIONS-> Fed

Entities: Fed, The Sydney Morning Herald, Chinese government, bond market, resource stocks, stimulus package, interest rates, consumer discretionary,
information technology, annual general meetings, Nasdaq, Atlanta Fed, Raphael Bostic, recession, Oil
</item 3>

<item 4>
Graph query row:
title: ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
article: ASX set for bright start despite Wall Street slip; $A drops
publisher: The Sydney Morning Herald
Federal_Reserve_entity: Federal Reserve
relationship: MENTIONS
connected_entity: $A
passage: ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Loading Perpetual climbed 6.3 per cent after it announced a strategic review of its three-pronged corporate structure that might see it demerge its corporate
trust and wealth management businesses and create a more focused asset manager. Shares in Endeavour lifted 2.4 per cent after it forecast earnings before
interest and tax in its hotels division would grow by more than $150 million over the next five years. The laggards Utilities and energy stocks were the weakest
performers on the local bourse, rising 0.6 per cent and 0.8 per cent, respectively.

Only five large-cap stocks declined, with Evolution Mining sinking a mammoth 13 per cent after it raised $525 million selling new shares to institutional
investors at $3.80 a share. The mining sector, however, managed to finish in positive territory.
</item 4>

<item 5>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Such cuts can act like steroids for financial markets. For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the
only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report. A separate report on Friday said growth
in US services industries, such as finance and construction, was weaker last month than economists expected. The report from the Institute for Supply Management
also suggested a slight easing in prices.

Excitement about a potentially easier Fed was more than enough to offset a fall for Apple, which is Wall Street’s most influential stock. The most valuable US
stock fell 0.5 per cent despite reporting stronger profit for the latest quarter than analysts expected. Analysts said investors were likely disappointed with
Apple’s forecast for revenue for the last three months of 2023.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> financial markets
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> financial markets
- There are no sacred numbers -MENTIONS-> financial markets
- Psagot worth 80% less since Rani Zim bought it -MENTIONS-> financial markets
- Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions -MENTIONS-> financial
markets

Entities: financial markets, Apple, Bank of America, Fed, The Sydney Morning Herald, US, ASX to rise after interest rate hopes boost US shares, Wall Street,
BofA Global Research, Institute for Supply Management, US services industries
</item 5>

<item 6>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
It had been above 5 per cent, and at its highest level in more than a decade, during October. The yield on the two-year Treasury, which more closely tracks
expectations for the Fed, went on a jagged run following the economic reports. It fell from 4.61 per cent just before the reports’ release to 4.57 per cent and
then yo-yoed before easing back to 4.57 per cent.

Loading Traders widely expect the Federal Reserve to hold its key interest rate steady at its meeting next week, before potentially cutting rates in March,
according to data from CME Group. Fed officials have recently hinted that the federal funds rate may indeed already be at its peak. It’s above 5.25 per cent, up
from nearly zero early last year. But Fed Chair Jerome Powell and others have also warned Wall Street about being overzealous in its predictions about how early
a cut could happen. Lower yields have been one reason prices for cryptocurrencies have been rising recently.
</item 6>

<item 7>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> $A
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> Utilities and energy stocks
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> Perpetual
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> Endeavour
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> Evolution Mining

Entities: $A, ASX set for bright start despite Wall Street slip; $A drops, US Treasury, Federal Reserve, The Sydney Morning Herald, stocks, bond market,
interest rates, recession, ASX, Wall Street, job market, inflation
</item 7>

<item 8>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> jobs
- Want to help your kids achieve their potential? This expert has some advice -MENTIONS-> jobs
- Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' -MENTIONS-> jobs
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> jobs
- Suleyman -SAID-> jobs

Entities: jobs, ASX set to open higher as Wall Street rebounds; $A rises, wages, ADP, ADP report, comprehensive jobs report, markets, overnight rate, private
employers, US government, Federal Reserve, The Age, interest rates, oil prices, ASX
</item 8>

<item 9>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> financial markets
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> financial markets
- There are no sacred numbers -MENTIONS-> financial markets
- Psagot worth 80% less since Rani Zim bought it -MENTIONS-> financial markets
- Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions -MENTIONS-> financial
markets

Entities: financial markets, financial system, Fed, The Sydney Morning Herald, US, economy, interest rates, ASX to rise after interest rate hopes boost US
shares, Expedia Group, Treasury yields, Katie Nixon, Northern Trust Wealth Management, investments, coronavirus crash, three high-profile US bank failures
</item 9>

<item 10>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
“Westpac’s results were better than expected, especially because it’s been a bit of a perennial underperformer in the past few years,” he said, adding that it
set the tone for results from NAB and ANZ, which are due to report in the next week. Loading Clark also said there was a reversal of the interest-rate trade
seen in recent weeks. “Long-term bond yields came off quite significantly in the last few days following weaker than expected employment data on Friday, which
has seen some investors swivel back into growth stocks,” he said. The market has now switched from pricing in an 80 per cent chance of a rate rise on Tuesday to
50-50, Clark said.

On Wall Street, the S&P 500 climbed 40.56 points on Friday, or 0.9 per cent, to 4,358.3.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -PUBLISHED_BY-> The Sydney Morning Herald
- ASX set to open lower as Wall Street closes September with more losses -PUBLISHED_BY-> The Sydney Morning Herald
- The best shows to stream in December -PUBLISHED_BY-> The Sydney Morning Herald
- McRae reveals hidden message he carried under collar of his grand final shirt -PUBLISHED_BY-> The Sydney Morning Herald
- ASX set to drop as Wall Street’s September slump deepens -PUBLISHED_BY-> The Sydney Morning Herald

Entities: The Sydney Morning Herald, ASX to rise after interest rate hopes boost US shares, US shares, Clark, Westpac, NAB, ANZ, interest-rate trade, long-term
bond yields, employment data, growth stocks, rate rise, S&P 500, Wall Street, ASX
</item 10>
=== END ITEM id=bfb1c4743371b23a ===
