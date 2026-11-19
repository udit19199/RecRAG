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

=== ITEM id=bf8bfcb6f7b49561 ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: Huang said this during cross-examination.
  Fact 0.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.2: Huang said Paradigm was told Bankman-Fried did not want investors as directors.
  Fact 0.3: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 1: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 1.0: The second week of the trial included standout testimony.
  Fact 1.1: The standout testimony came from Caroline Ellison.
  Fact 1.2: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.3: Caroline Ellison claimed she took $14 billion from customers.
  Fact 1.4: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 1.5: Caroline Ellison claimed she used the money as a line of credit.
  Fact 1.6: Caroline Ellison claimed she did this under the instruction of SBF.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried trial, wire fraud, Ellison, FTX, Sam Bankman-Fried,
conspiracy to take FTX customer funds, fake balance sheets, How is it still getting worse for Sam Bankman-Fried?, Genesis, The Verge
</item 1>

<item 2>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried didn’t ask where the $8 billion went, risk management,
fiat@ftx, Sam Bankman-Fried trial, wire fraud, conspiracy, Congress, FTX, Sam Bankman-Fried, fraud, The Verge
</item 2>

<item 3>
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
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Damian Williams
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The best new TV series on every streaming service -MENTIONS-> Ponzi scheme
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison

Entities: Ponzi scheme, crypto markets, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand,
Alameda, Ellison, Cohen, Fortune, FTX, Sam Bankman-Fried
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, FTT, Ellison, FTX, Wang, Sam Bankman-Fried, Binance, The Verge
</item 5>

<item 6>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, theft, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand,
Alameda Research, Rehn, Fortune, FTX, Sam Bankman-Fried, government
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Damian Williams
- Damian Williams -SAID-> financial fraud
- Damian Williams -PARTICIPATED_IN-> Press conference
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto

Entities: Damian Williams, crypto, alleged crimes, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take
the stand, Fortune, FTX, the government, Sam Bankman-Fried, the trial, MIT, fraud
</item 7>

<item 8>
Graph query row:
publisher: TechCrunch
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
published_on: 2023-10-06T17:22:02+00:00
FTX_entity: FTX Ventures
discussed_entity: defrauding customers
Sam_Bankman_Fried: Sam Bankman-Fried
passage: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 8>

<item 9>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
Meanwhile, Bankman-Fried tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in treasuries.”

Pointing at Bankman-Fried, Rehn said, “This man stole billions of dollars from thousands of people.”

So how was the defense going to follow it up? I was very curious, having learned yesterday that Bankman-Fried had never been offered a plea deal since he and
his attorneys had told the government they wouldn’t negotiate. Surely there would be some manner of evidence, some something, that would have made him so
confident.

There was, instead, a metaphor.

Defense attorney Mark Cohen, with the energy of a patient father telling his obnoxious children a bedtime story, assured us that working at a startup was like
building a plane while flying it, and that FTX the plane had flown right into the perfect storm: the crypto crash.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Mark Cohen
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Mark Cohen
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Mark Cohen
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Mark Cohen
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Mark Cohen

Entities: Mark Cohen, Is Sam Bankman-Fried’s defense even trying to win?, Rehn, crypto crash, customer assets, FTX, the government, Sam Bankman-Fried, The Verge
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=bf8bfcb6f7b49561 ===

=== ITEM id=fa1536710dc37fd8 ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: The parties referred to as 'they' get many other benefits from Android.
  Fact 0.1: Google could solve this problem today if Google wanted to.
Sentence 1: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 1.0: The case was filed by Helena World Chronicle.
  Fact 1.1: Helena World Chronicle is an Arkansas-based publisher.
  Fact 1.2: The case argues that Google siphons off news publishers' content.
  Fact 1.3: The case argues that Google siphons off news publishers' readers.
  Fact 1.4: The case argues that Google siphons off news publishers' ad revenue.
  Fact 1.5: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 1>

<item 2>
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
Graph query row:
article: ChatGPT: Everything you need to know about the AI-powered chatbot
published_on: 2023-09-28T20:03:39+00:00
publisher: TechCrunch
Google_entity: Google DeepMind
related_topic: Google Play
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 7>

<item 8>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He speculated that the company didn’t want the public to think too hard about how search works because that awareness “encourages regulators and makes people
realize, ‘Oh, there’s a lot of money here.’”

There has always been advertising and polemics from cranks, scammers, and liars. But now we see this stuff surfacing alongside truth, and we can’t tell the
difference. We move through our lives with a greater sense of distrust and fear and insecurity. At the end of the day, it’s Google’s world, and the SEOs are
only living in it.

And as much as I might hate the way the SEOs who don’t follow Google’s rules have altered my online experience, the reality is that most people running a
company will break whatever rules they are able to get away with breaking.

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> internet
- ChatGPT can now browse the internet (again) -MENTIONS-> internet
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> internet
- 6 VCs explain how startups can capture and defend marketshare in the AI era -MENTIONS-> internet
- California’s privacy watchdog eyes AI rules with opt-out and access rights -MENTIONS-> internet

Entities: internet, The people who ruined the internet, SEOs, regulators, polemics, Google, search, advertising, The Verge
</item 8>

<item 9>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
Although the situation remains precarious, connectivity eventually was partially restored. There are indications that the U.S. government is putting pressure on
Israel to restore internet connectivity in Gaza.

Meanwhile in India, Apple has notified several prominent Indian lawmakers and journalists of potential state-sponsored attacks on their iPhones, just months
ahead of the country’s general elections. The individuals targeted include opposition leader Rahul Gandhi, key Congress party figure Shashi Tharoor, and leaders
from several other influential political parties.

Atlassian issued a warning about a critical security flaw that could result in significant data loss for its customers. Notably, this product was recently the
target of Chinese state-sponsored hackers. Atlassian has not yet detailed how the flaw can lead to data loss but has urged customers to patch against the flaw
immediately.

There’s more:

About flippin’ time, too: Apple has recently fixed a longstanding vulnerability in its iOS software that undermined a privacy feature. Security researchers
found that the feature, introduced in iOS 14, was not working as intended.

Graph facts:
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Atlassian
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Atlassian
- Atlassian -PARTICIPATED_IN-> Atlassian layoffs
- Atlassian -OWNS-> Loom
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Atlassian, Apple, TechCrunch, India, If your startup doesn’t seem impossible at first, it’s not hard enough, Indian lawmakers and journalists,
potential state-sponsored attacks on iPhones, general elections, Rahul Gandhi, Congress party, Shashi Tharoor, critical security flaw, data loss, Chinese
state-sponsored hackers, longstanding vulnerability
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=fa1536710dc37fd8 ===

=== ITEM id=fa9e38fb12276731 ===
Gold sentences and facts:
Sentence 0: James is seeking $250 million in penalties and a ban on Trump doing business in New York, his home state.
  Fact 0.0: James is seeking $250 million in penalties.
  Fact 0.1: James is seeking a ban on Trump doing business in New York.
  Fact 0.2: New York is Trump's home state.
Sentence 1: No apartment in New York City has ever sold for close to that amount, James said.
  Fact 1.0: No apartment in New York City has ever sold for close to that amount, according to James.
Sentence 2: The prosecution argues that was to mask a drop in the value of one of his other properties.
  Fact 2.0: The prosecution argues that a transaction was meant to mask a drop in the value of one of Trump's other properties.

Retrieved items:
<item 1>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
James, a Democrat, sued Trump and the Trump Organization a year ago, alleging a pattern of duplicity that she dubbed “the art of the steal,” a twist on the
title of Trump’s 1987 business memoir “The Art of the Deal.”

The lawsuit accused Trump and his company of routinely inflating the value of assets like skyscrapers, golf courses and his Mar-a-Lago estate in Florida,
padding his bottom line by billions.

Among the allegations were that Trump claimed his Trump Tower apartment in Manhattan — a three-story penthouse replete with gold-plated fixtures — was nearly
three times its actual size and valued the property at $327 million. No apartment in New York City has ever sold for close to that amount, James said.

Trump valued Mar-a-Lago as high as $739 million — more than 10 times a more reasonable estimate of its worth.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> New York City
- The people who ruined the internet -MENTIONS-> New York City
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> New York City
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> New York City
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> New York City

Entities: New York City, Fortune, lawsuit, Manhattan, fraud, Donald Trump, Trump Organization, The Art of the Deal, inflating asset values, Mar-a-Lago, Florida,
Trump Tower apartment, James, Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former
president
</item 1>

<item 2>
Attorney general going after Jeffrey Epstein's estate says she was fired for her dogged pursuit: 'My bar license, my integrity were more important to me'
(Fortune, 2023-11-17T21:59:14+00:00)
The place is empty. On the silver chain round her neck hangs a gift from her husband: a pendant in the shape of the scales of justice.

Epstein’s estate was the first to settle with the Virgin Islands. A year ago, it agreed to pay $105 million, plus half the proceeds from the sale of Little St.
James, AKA “Little St. Jeff,” the private island where authorities allege some of Epstein’s crimes took place.

Then, last January, Wall Street billionaire Leon Black settled for $62.5 million. Next came JPMorgan Chase., in September, for $75 million. None of the parties
admitted to wrongdoing. All of them declined to comment for this story.

Without those cases, the world may have never glimpsed some of the secrets Epstein and his circle harbored, or the depths of his influence here in the
Caribbean.

Graph facts:
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> JPMorgan Chase
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> JPMorgan Chase
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-MENTIONS-> JPMorgan Chase
- Jamie Dimon -WORKS_FOR-> JPMorgan Chase
- JPMorgan Chase -ACCUSED_OF-> violating federal laws

Entities: JPMorgan Chase, Fortune, Caribbean, Jeffrey Epstein, Attorney general, Epstein’s estate, Little St. James, Leon Black, Epstein’s crimes, Epstein
estate settlement with the Virgin Islands, Leon Black settlement, JPMorgan Chase settlement, Attorney general going after Jeffrey Epstein's estate says she was
fired for her dogged pursuit: 'My bar license, my integrity were more important to me', Virgin Islands
</item 2>

<item 3>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
Engoron’s ruling, in a phase of the case known as summary judgment, resolves the key claim in James’ lawsuit, but six others remain.

Engoron is slated to hold a non-jury trial starting Oct. 2 before deciding on those claims and any punishments he may impose. James is seeking $250 million in
penalties and a ban on Trump doing business in New York, his home state. The trial could last into December, Engoron has said.

Trump’s lawyers had asked the judge to throw out the case, which he denied. They contend that James wasn’t legally allowed to file the lawsuit because there
isn’t any evidence that the public was harmed by Trump’s actions. They also argued that many of the allegations in the lawsuit were barred by the statute of
limitations.

Engoron, noting that he had “emphatically rejected” those arguments earlier in the case, equated them to the “time-loop in the film ‘Groundhog Day.'”

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, Fortune, defrauding banks, Engoron, non-jury trial, Groundhog Day, statute of limitations, public harm, trial could last into December,
Donald Trump, banks, James, James’ lawsuit, Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation
against the former president
</item 3>

<item 4>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
They haven’t provided a more recent estimate.

Space inflation wasn’t the only issue. Between 2014 and 2015, Trump increased the value of the apartment to $US327 million from $US200 million. The prosecution
argues that was to mask a drop in the value of one of his other properties. “A discrepancy of this order of magnitude, by a real estate developer sizing up his
own living space of decades, can only be considered fraud,” Judge Arthur Engoron wrote in a September ruling. ‘The company has never been stronger and never
been better.’ Eric Trump Bloomberg values the property using recent Trump Tower sales, as well as those of comparable Manhattan penthouses. While some splashy
residences in much newer buildings on so-called Billionaires Row have listed for well in excess of $US100 million, they typically boast higher ceilings, more
up-to-date finishes, contemporary layouts, outdoor space and better views.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Manhattan, Arthur Engoron, fraud, Donald Trump, The $777 million surprise: Donald Trump is getting richer, Trump Tower, Eric Trump,
Billionaires Row, Bloomberg
</item 4>

<item 5>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
A judge ruled Tuesday that Donald Trump committed fraud for years while building the real estate empire that catapulted him to fame and the White House.

Judge Arthur Engoron, ruling in a civil lawsuit brought by New York’s attorney general, found that the former president and his company deceived banks, insurers
and others by massively overvaluing his assets and exaggerating his net worth on paperwork used in making deals and securing financing.

Engoron ordered that some of Trump’s business licenses be rescinded as punishment and said he would continue to have an independent monitor oversee the Trump
Organization’s operations.

A Trump spokesperson did not immediately respond to a request for comment on the ruling. Trump has long insisted he did nothing wrong.

The decision, days before the start of a non-jury trial in Attorney General Letitia James’ lawsuit, is the strongest repudiation yet of Trump’s carefully
coiffed image as a wealthy and shrewd real estate mogul turned political powerhouse.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, real estate, Fortune, White House, Arthur Engoron, Letitia James, fraud, civil lawsuit, non-jury trial, Donald Trump, Trump Organization,
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president
</item 5>

<item 6>
Attorney general going after Jeffrey Epstein's estate says she was fired for her dogged pursuit: 'My bar license, my integrity were more important to me'
(Fortune, 2023-11-17T21:59:14+00:00)
In the end the estate returned $80 million in tax incentives, agreed to pay half a million to remediate environmental damage to one of Epstein’s islands,
committed half the proceeds from the future sale of Little St James and established a victims’ compensation fund.

Officially, George, not the governor, was in charge of making the legal calls. But Bryan was still her boss and held the purse strings. It was a delicate
balance, George said.

Trying to untangle a half-billion-dollar estate is a formidable task under the best of circumstances. It was even harder given the code of silence still
surrounding the dead financier. Many of Epstein’s employees had signed nondisclosure agreements. Some alerted the estate’s lawyers the moment investigators
reached out. On Little St. Jeff, groundskeepers avoided eye contact when authorities approached.

George herself traveled to Little St. Jeff. She still can’t shake eerie absurdity of seeing a dentist’s chair inside the luxury island villa.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> the governor
- the governor -SAID-> Kentucky sports betting
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune

Entities: the governor, Fortune, Jeffrey Epstein's estate, Jeffrey Epstein, Little St. Jeff, Attorney general going after Jeffrey Epstein's estate says she was
fired for her dogged pursuit: 'My bar license, my integrity were more important to me', Bryan, George, Little St James, luxury island villa, dentist's chair,
tax incentives, environmental damage, victims' compensation fund, nondisclosure agreements
</item 6>

<item 7>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
The company was fined $1.6 million. One of the executives, Trump’s longtime finance chief Allen Weisselberg, pleaded guilty and served five months in jail. He
is a defendant in James’ lawsuit and gave sworn deposition testimony for the case in May.

James’ lawsuit does not carry the potential of prison time, but could complicate his ability to transact real estate deals. It could also stain his legacy as a
developer.

James has asked Engoron to ban Trump and his three eldest children from ever again running a company based New York. She also wants Trump and the Trump
Organization barred from entering into commercial real estate acquisitions for five years, among other sanctions. The $250 million in penalties she is seeking
is the estimated worth of benefits derived from the alleged fraud, she said.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, real estate, Fortune, fraud, Engoron, Donald Trump, Trump Organization, Allen Weisselberg, James, James’ lawsuit, Donald Trump defrauded
banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president
</item 7>

<item 8>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
The following year, Trump valued them at $US4.1 million each — more than 65 times greater. Trump has continued to value stabilised apartments in the building at
market rates, including in his 2021 statement of financial condition. Rent-stabilised units “with a positive cash flow are often valued at 15-25 cents a dollar
of the open market value, versus single digits for negative cash flow,” said Jonathan Miller, president of appraiser Miller Samuel. Still, Trump’s optimism was
shown to be at least partially justified. By 2021, only five of the 12 apartments remained rent stabilised. The prosecution also claims two additional
non-stabilised apartments in the building were overvalued because Trump ignored an option his daughter, Ivanka Trump, had to buy them at a price lower than the
market value he used.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, the prosecution, Donald Trump, The $777 million surprise: Donald Trump is getting richer, Jonathan Miller, Miller Samuel, Ivanka Trump,
rent-stabilised unit valuation, apartment overvaluation
</item 8>

<item 9>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
Beyond mere bragging about his riches, Trump, his company and key executives repeatedly lied about them on his annual financial statements, reaping rewards such
as favorable loan terms and lower insurance premiums, Engoron found.

Those tactics crossed a line and violated the law, the judge said, rejecting Trump’s contention that a disclaimer on the financial statements absolved him of
any wrongdoing.

“In defendants’ world: rent regulated apartments are worth the same as unregulated apartments; restricted land is worth the same as unrestricted land;
restrictions can evaporate into thin air; a disclaimer by one party casting responsibility on another party exonerates the other party’s lies,” Engoron wrote in
his 35-page ruling. “That is a fantasy world, not the real world.”

Manhattan prosecutors had looked into bringing a criminal case over the same conduct but declined to do so, leaving James to sue Trump and seek penalties that
could disrupt his and his family’s ability to do business in the state.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Manhattan, Trump's company, key executives, Manhattan prosecutors, defrauding banks, lying on annual financial statements, disclaimer
absolving Trump of wrongdoing, tactics crossing a line and violating the law, criminal case over the same conduct, Trump's family, Engoron, Donald Trump, banks,
James
</item 9>

<item 10>
Graph query row:
source: Fortune
article: Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images'
published_on: 2023-10-10T20:17:09+00:00
matched_entity: election defeat of Donald Trump
connected_entities: ['New York City', 'Vietnam War', 'Supreme Court', 'baby boomers', 'millennials', 'internet', 'activist', 'Ukraine', 'mental health', 'Gen
Z', 'worst war in Israel in 50 years', 'Hamas attack on Israel', 'Israeli civilians', 'Palestinian civilians', 'once-per-century pandemic', 'Jan. 6
insurrection', '6 insurrection', 'European ground war in Ukraine', 'Market crash in 2008', 'Market crash in 2020', 'Great Depression', 'workplace attitudes',
'financial habits', 'young workers ages 18 to 26', '2023 Edelman report', 'Gen Xers', 'older generations', 'war in Israel', 'George Floyd', 'murder of George
Floyd', 'Black Lives Matter', 'Black Lives Matter protests and riots', 'armed right-wing extremists', 'the Capitol', 'storming the Capitol', 'election defeat of
Donald Trump', 'school shootings', 'Columbine shooting', 'Parkland shooting', 'American Psychological Association', 'disturbing images', 'human-caused global
warming', 'monthlong heat wave', 'fires across the Mediterranean', 'Canadian wildfires', 'Northeast', 'Arctic', 'ice melt', 'Greta Thunberg', 'Speaking at the
United Nations', 'mass extinction', 'Edelman report', 'environmental, social, and governance (ESG)', 'work-life balance', 'fair pay', 'global pandemic', 'two
recessions', 'student debt crisis', 'Bankrate', 'Dayo Abinusawa', 'Awa Business School', 'Cambridge University’s Judge Business School', 'Fidelity', '18- to
35-year-olds', 'saving', 'money', 'careers', 'psychological scars', 'Millennials', '9/11 terrorist attacks', 'U.S. invasion of Iraq', 'Iraq', 'Baby boomers',
'civil rights movement', 'teenage TikTok users', 'K-pop fans', 'Trump campaign', 'Trump campaign rally', 'Roe v. Wade', 'abortion funds', 'Abinusawa', 'a
society where the young have little to no hope for the future is not a sustainable one', 'workplace decision-making positions', 'wealth', 'Hamas', 'Arizona',
'American politics', 'Pew Research', 'carbon emissions', 'TikTok', 'sustainability', 'Israel', 'climate change', 'Mediterranean', 'Matt Gaetz', 'social issues',
'London', 'inflation', 'U.S.', 'Donald Trump', 'Florida', 'World War II', 'social media', 'United Nations']
</item 10>
=== END ITEM id=fa9e38fb12276731 ===
