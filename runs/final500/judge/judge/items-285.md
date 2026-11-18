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

=== ITEM id=586790e06b43ea69 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader.
  Fact 0.1: Bankman-Fried said it was okay to continue withdrawing money from the account.
  Fact 0.2: The withdrawals were okay as long as the amount was less than FTX's total trading revenue.
Sentence 1: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry.
  Fact 1.0: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto.
  Fact 1.1: Bankman-Fried made himself out to be the trustworthy face of a sometimes-shady industry.
Sentence 2: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 2.0: Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance.
  Fact 2.1: Bankman-Fried admitted he had been told about that money.
  Fact 2.2: Bankman-Fried admitted this only after Judge Lewis Kaplan intervened.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense countered that the FTX founder acted in good faith.
  Fact 3.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 3.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 1>

<item 2>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 2>

<item 3>
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
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 5>

<item 6>
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
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 8>

<item 9>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 9>

<item 10>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 10>
=== END ITEM id=586790e06b43ea69 ===

=== ITEM id=30c454822f7f2642 ===
Gold sentences and facts:
Sentence 0: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 0.0: The company did not release the full model, Gemini Ultra.
  Fact 0.1: The company released only a 'lite' version called Gemini Pro.
Sentence 1: “Certainly there wasn’t a valid alternative to Google at the time,” Cue said.
  Fact 1.0: Cue said there certainly wasn't a valid alternative to Google at the time.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Cue is one of the highest-profile witnesses in the case so far, in part because the deal between Google and Apple — which makes Google the default search engine
on all Apple devices and pays Apple billions of dollars a year — is central to the US Department of Justice’s case against Google.

Cue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether those two statements can be simultaneously true
became the question of the day.

Apple is in court because of something called the Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s
products. The ISA has been in place since 2002, but Cue was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In
testimony today, the Justice Department grilled Cue about the specifics of the deal.

When the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward the
search engine.
</item 1>

<item 2>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
relationship: PUBLISHED_BY
related: TechCrunch
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 4>

<item 5>
Graph query row:
article: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’
relationship: PUBLISHED_BY
related: The Verge
</item 5>

<item 6>
Graph query row:
article: Early impressions of Google’s Gemini aren’t great
relationship: PUBLISHED_BY
related: TechCrunch
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 10>
=== END ITEM id=30c454822f7f2642 ===

=== ITEM id=a5d8390e6f6b51c6 ===
Gold sentences and facts:
Sentence 0: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 0.0: A woman referred to in the quote has the Eras Tour booked all through next year.
  Fact 0.1: A woman referred to in the quote would rather die than cancel a show.
Sentence 1: In July, speaking on the podcast he shares with his brother, New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City.
  Fact 1.0: Kelce spoke in July on the podcast he shares with his brother, New Heights.
  Fact 1.1: Kelce said he intended to give Swift a friendship bracelet with his number on it.
  Fact 1.2: Kelce intended to give the friendship bracelet during Swift's Eras Tour concert in Kansas City.
Sentence 2: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 2.0: The game is taking place at Arrowhead Stadium.
  Fact 2.1: Swift has performed at Arrowhead Stadium.
  Fact 2.2: Swift has attended a game at Arrowhead Stadium before.
Sentence 3: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 3.0: Swift has shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Well I would say that Travis Kelce's had a lot of big catches in his career. This would be the biggest."

Sept. 26: Chiefs games get even more popular

"Welcome to New York (*New Jersey), it's been waiting for you." - Welcome to New York

The defending Super Bowl champions are already one of the more popular teams in the league, but it is becoming the hottest ticket in town, no matter what town
that is.

"Since Swift's appearance at the Chiefs game on Sunday, the Jets have sold more tickets for their game against the Chiefs in a single day than since the season
started -- by more than double," a representatives at StubHub told to Town & Country magazine.

Sept. 27: Mahomes confirms he met Swift

"It was enchanting to meet you, all I can say is I was enchanted to meet you."
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But at the same time, they're overdoing it a little bit."

Jason also chimed in on the matter, saying, "I just think the NFL is not used to celebrities coming to the games. Like basketball has it all figured out.
They're all courtside, they're sitting there. They show 'em once or twice, but then they get back to the game."

Oct. 12: Swift attends Chiefs-Broncos

"Now it's too late for you and your white horse *Bronco, to come around" - White Horse

One day after her movie premiere, Swift is in the house for the Chiefs' "Thursday Night Football" divisional matchup against the Broncos. The game is taking
place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will have minimal Swift coverage,
though so far there has already been a mention of the superstar.

Oct.
</item 2>

<item 3>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 3>

<item 4>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
How do you not get that? That's called images right there."

Oct. 1: Swift attends Chiefs-Jets

"Welcome to New York, it's been waiting for you," - Welcome to New York

OK, so the Jets technically play in New Jersey, but we're just basing this off their name. Swift went to her second consecutive Chiefs game, showing up with
some famous friends, Blake Lively, Ryan Reynolds and Hugh Jackman. In the week leading up to the game, there were reports that the singer would be at the game.
The Chiefs defeated the Jets 23-20.

Oct. 5: NFL defends itself

Swift coverage is everywhere and the league even changed their social media bio to one of the singer's lyrics. The league's X (formerly Twitter) bio to read,
"NFL (Taylor's Version)," while their Instagram account read, "Chiefs are 2-0 as Swifties."
</item 5>

<item 6>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 6>

<item 7>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
“OF ALL 230 SONGS IN THE DISCOGRAPHY I’M SICK,” one fan commented, while another person wrote: “It’s unbelievable how invested I am in this.”

“I’m gonna need to go back and see all photos of him from the last three months and dissect if he’s wearing friendship bracelets,” a third wrote.

In other photos from the charity event - which were shared to Instagram by Jordan’s KC photography - Kelce wasn’t seen wearing the friendship bracelet, leading
many people to believe that he received the bracelet from a fan that day.

Two days after his charity event, the “All Too Well” singer was spotted cheering for the Kansas City Chiefs tight end from the box seats at Arrowhead Stadium in
Kansas City, Missouri. Not only was she seen sitting next to Kelce’s mother, Donna, but she also appeared to yell “let’s f***ing go” in response to Kelce’s
third-quarter touchdown reception.
</item 7>

<item 8>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
The dating speculation first began in July, when Kelce confessed that he tried and failed to give Swift a friendship bracelet with his number on it during her
Eras Tour show. On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24
September. Since then, Swift has attended three more of Kelce’s games.

Kelce showed his support for Swift when he attended her Eras Tour show in Buenos Aires, Argentina, on 11 November. During the concert, Swift gave a shout-out to
the Chiefs player when she changed her “Karma” lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang,
letting out a laugh as fans in the audience screamed in response. The “All Too Well” singer was later seen running into his arms and kissing the NFL player
following the concert, as he whisked her away backstage.
</item 8>

<item 9>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I appreciate you always trying to make it to both of our games.”

“She also told me thank you because she was in an alternate universe, she was sitting there with Deadpooland Wolverine,” Kelce added, referencing Reynolds’ and
Jackman’s respective superhero movie characters. “It was a hot ticket, man, you know what I’m saying? That suite was rockin’.

“They were enjoying themselves. Somebody told me that everybody was enjoying themselves in that suite,” Kelce said.

Not only has NFL viewership increased since Swift began attending her rumoured boyfriend’s games, but jersey sales for the Chiefs tight end have reportedly
spiked by 400 per cent, according to a report from the NFL’s e-commerce partner, Fanatics.

Following critcism the league is focusing too much on Swift, the NFL responded with an official statement handed to People on 4 October.

“We frequently change our bios and profile imagery based on what’s happening in and around our games, as well as culturally,” the message read.
</item 9>

<item 10>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.
</item 10>
=== END ITEM id=a5d8390e6f6b51c6 ===

=== ITEM id=8f32d682c3dafe03 ===
Gold sentences and facts:
Sentence 0: They’d all have significant value if president of baseball operations Jerry Dipoto decided to move one for a bat.
  Fact 0.0: Jerry Dipoto is president of baseball operations.
  Fact 0.1: Those referred to as 'they' would all have significant value if Jerry Dipoto decided to move one for a bat.
Sentence 1: Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.
  Fact 1.0: Aaron Judge came away with a new contract.
  Fact 1.1: Justin Verlander came away with a new contract.
  Fact 1.2: Trea Turner came away with a new contract.
  Fact 1.3: Xander Bogaerts came away with a new contract.

Retrieved items:
<item 1>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
While the offseason has gotten rolling, trade season hasn’t quite arrived. The Tigers acquired Mark Canha from Milwaukee in the biggest swap of the first couple
weeks. Otherwise, the GM Meetings simply served to set the stage for future transactions.

That’ll surely involve some notable trades. Unlike the summer deadline, projecting which players will change hands is difficult at this stage of the offseason.

Aside from the A’s, there aren’t any teams clearly in rebuild mode. Clubs like the Rockies, Royals, White Sox, Nationals and Angels have uphill paths to
contention, yet none has indicated they’re prepared to punt on the 2024 season entirely.

The Padres, Twins, Rays, Brewers and Guardians are all aiming to contend but have questions about their payroll. That’s standard operating procedure for Tampa
Bay, Milwaukee and Cleveland. San Diego and, to a lesser extent, Minnesota have shown a willingness to spend in recent years but are scaling back — in both
cases, likely tied to their uncertain local television rights contracts.
</item 1>

<item 2>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Twins could trade Max Kepler, Jorge Polance

Date: Sunday, Dec. 17

Source: Jon Morosi, MLB Network

The Twins are coming off an AL Central title, but could be looking to deal away two of their key players form the team. Max Kepler and Jorge Polanco could both
reportedly be on the trade market, per MLB Network's Jon Morosi, though their market could hinge on the status of other infield free agents.

Kepler appeared in 130 games for the Twins, and posted an impressive .260/.332/.484 slash line with 24 homers for Minnesota. Polanco was moved to an infield
utility role with Edouard Julien and Royce Lewis emerging into regulars, but was still solid in 80 games, logging a .255/.335/.454 slash with 14 homers.

Yamamoto requests meeting with Yankees

Date: Sunday, Dec.
</item 2>

<item 3>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
MLB's winter meetings are always a prime time for moves to happen as many of the top executives, players and agents all come together at the same spot.

Last year's winter meetings wound up being explosive. Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.

This year's winter meetings could wind up being similarly busy. There is a growing sense Shohei Ohtani could sign during the winter meetings, and a move of that
magnitude could kick off the next wave of moves. There's also a number of top players on the trade market, including star Padres outfielder Juan Soto.

Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.

WINTER MEETINGS: Shohei Ohtani rumors, Juan Soto trade destination come to forefront

Sporting News will be tracking all the latest news and rumors from the 2023 Winter Meetings.
</item 3>

<item 4>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Verdugo had been the Red Sox's starting outfielder for the last four seasons, coming over as the centerpiece in the trade that sent Mookie Betts to the Dodgers.
Verdugo hit .281 with 43 home runs, 206 RBI and 16 stolen bases over 2,071 plate appearances with Boston. He also led the American League in outfield assists in
2020 and 2023.

In return, Boston will get Weissert and two pitching prospects headlined by Fitts — the Yankees' No. 4-ranked pitching prospect and 12-ranked prospect overall.

Relievers Yates, Devenski get one-year deals for 2024

Date: Tuesday, Dec. 5

Source: Ken Rosenthal, The Athletic and Jeff Passan, ESPN

Veteran relievers Kirby Yates and Chris Devenski know where they'll be pitching in 2024, according to reports Tuesday.
</item 4>

<item 5>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Davis, Mike Yastrzemski

Guardians: Josh Naylor, Cal Quantrill

Mariners: Ty France

Marlins: Josh Bell

Mets: Jeff McNeil

Nationals: Hunter Harvey, Lane Thomas

Orioles: Anthony Santander

Padres: Scott Barlow

Phillies: Nick Castellanos

Rays: Manuel Margot, Harold Ramírez

Red Sox: Kenley Jansen, Chris Martin, Nick Pivetta

Rockies: Brendan Rodgers

Royals: Salvador Perez

Tigers: Spencer Turnbull

Twins: Kyle Farmer
</item 5>

<item 6>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
It also makes him less likely to be traded this offseason.

First-year GM Chris Getz has suggested no one on the roster is truly untouchable, an understandable approach for a team that just lost 101 games. Neither owner
Jerry Reinsdorf nor Getz has called this a rebuild, though.

Any slim hopes the White Sox may have of competing in 2024 would be dashed by trading Cease, the only above-average starting pitcher on the roster.

7. Alex Verdugo, RF, Red Sox

Verdugo’s name has been floated in trade rumors for a few seasons. The Red Sox have held onto him thus far. They have a new baseball operations leader in Craig
Breslow, who has acknowledged the team has gotten early calls on the left-handed hitting outfielder.

Jarren Duran and Masataka Yoshida also hit from the left side. Dealing Verdugo could free some at-bats for a right-handed addition while perhaps bringing in
immediate help for the pitching staff or at second base.
</item 6>

<item 7>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
3

Source: Susan Slusser, San Francisco Chronicle

The Giants have missed out on big names in the past like Aaron Judge and Carlos Correa, but they are looking to turn that around in 2023 with heavy involvement
and Ohtani, among others. Per San Francisco Chronicle's Susan Slusser, teams believe the Giants "might have an edge" in the pursuit of Japanese ace Yoshinobu
Yamamoto.

Dylan Cease trade market heating up

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

One of the biggest names on the market early is White Sox ace Dylan Cease in trade talks. MLB Network's Jon Morosi reported three teams, the Dodgers, Braves and
Orioles, are among several involved discussing a deal for Cease as the winter meetings begin.

Rays, Mariners talking trades in winter meetings

Date: Sunday, Dec.
</item 7>

<item 8>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
His 3.39 mark across 193 2/3 innings this past season wasn’t quite so dominant but is still high-end production.

Burnes struck out just under 26% of opposing hitters. While his lowest mark since his rookie season, that’s still an above-average figure.

Adames is coming off a less impressive year. The 28-year-old hit .217/.310/.407 through 638 trips to the plate. That’s slightly below-average offense, although
he still connected on 24 home runs.

Adames is an excellent defensive shortstop with plus power in an offseason where there are essentially no free agent shortstops of note. Even with a mediocre
on-base percentage, he’d generate plenty of interest.

4. Shane Bieber, SP, Guardians

Develop high-end starting pitching, trade a top starter as they become more expensive, replace him with talented younger arms. The Guardians have trodden this
path with Corey Kluber, Mike Clevinger and Carlos Carrasco.
</item 8>

<item 9>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
16. Mariners SP

We initially had Bryan Woo in this spot. It could just as easily have been Bryce Miller or Emerson Hancock, so we’ll cheat and just mention the Seattle rotation
in general.

They’re almost certainly not going to trade George Kirby. It’d take a lot to pry away Logan Gilbert. Moving any of Woo, Miller or Hancock could be more
appealing for a front office that is looking for ways to add long-term offensive upside.

None of that trio has even reached one year of major league service. Hancock only has three big league starts under his belt. Miller and Woo spent more time in
the rotation this year, starting 25 and 18 games respectively. They both turned in mid-rotation results.

Other clubs probably have differing opinions on which right-hander they prefer. (The Mariners likely have their own slight preferences.) They’d all have
significant value if president of baseball operations Jerry Dipoto decided to move one for a bat.

17.
</item 9>

<item 10>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
New York has subsequently walked back some of their statements about a possible retool. They’re still aiming to compete in ’24 even if they’re signaling they
won’t blow away the top of the free agent market.

First-year president of baseball operations David Stearns has twice publicly said he anticipates Alonso being the Opening Day first baseman in Queens. That
won’t stop teams from trying, but an offseason deal looks decidedly unlikely.

Others of note

A’s: Seth Brown

Braves: Vaughn Grissom

Brewers: Adrian Houser, Devin Williams, Brandon Woodruff

Cardinals: Alec Burleson, Tommy Edman, Nolan Gorman, Tyler O’Neill

Giants: J.D.
</item 10>
=== END ITEM id=8f32d682c3dafe03 ===

=== ITEM id=28d321c4422ee265 ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made these minor physical upgrades alongside the big improvements.
  Fact 0.2: Valve's new iteration includes big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units of the Steam Deck OLED ready to ship on that same day.

Retrieved items:
<item 1>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 1>

<item 2>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 3>

<item 4>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 4>

<item 5>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 5>

<item 6>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 6>

<item 7>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 7>

<item 8>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 8>

<item 9>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 9>

<item 10>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Valve will be selling three models:

$649 - 1 TB OLED (up from 512 GB)

$529 - 512 GB OLED (up from 256 GB)

$399 - 256 GB non-OLED Steam Deck (up from 64 GB)

Plus, for a limited time, the company will offer a special edition 1 TB Steam Deck OLED made with a translucent shell for $679 — only in the U.S. and Canada.
Though let’s be real: This thing will sell like hotcakes and Valve will reconsider the definition of “limited supply.” Have I considered ordering one of these
limited editions? Yes. But I am also practicing restraint. Like, how many Steam Decks should one human have?

I suppose that’s the question, really. If you already have a Steam Deck, is the OLED model technically an incremental improvement? Yes. But you might find that
the screen and the battery justify the double dip.
</item 10>
=== END ITEM id=28d321c4422ee265 ===

=== ITEM id=0e9d47655a0ee39d ===
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
</item 1>

<item 2>
Graph query row:
article: NBA Rookie of the Year Odds - Wembanyama the early favorite
published_on: 2023-10-04T22:54:23+00:00
connection: PUBLISHED_BY
outlet: Sporting News
sportsbook: Caesars Sportsbook
</item 2>

<item 3>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
Again, we'll keep an eye out for any news regarding ESPN Bet Sportsbook VT.

Other potential Vermont sportsbooks

Vermont's new sports betting law allows for as many as six online sportsbooks, and while we know of a few of these we're still waiting to find out which ones go
live in the state. Here are a couple of other online sports betting sites that are thriving in other states, one or both of which could potentially come to
Vermont eventually.

Caesars Vermont promo code

Caesars Sportsbook has become one of the most popular online sportsbooks in the US in several states, and it should prove a favorite in Vermont as well. We
expect Caesars to offer the same offer in VT is has elsewhere, with new sign-ups eligible to receive up to a $1,000 Bonus Bet if their first bet loses.

Caesars covers wide range of markets and different bet types while also featuring a terrific live betting platform.
</item 3>

<item 4>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
11, 2024. BetMGM and ESPN Bet are also among the applicants, making them possible candidates to launch in Vermont in the new year as well. Here are details
regarding those five online sportsbooks and their welcome offers.

FanDuel Vermont promo code

Vermont already knows FanDuel thanks to its daily fantasy sports site. FanDuel Sportsbook is one of the most popular across the US, and FanDuel has already
applied for a license to launch in Vermont. FanDuel has a great welcome offer inviting new sign-ups simply to make an initial $5 wager and get back $200 in
Bonus Bets.

Meanwhile, for Vermont bettors FanDuel is running an Early Sign Up Special that gets you an extra $100 in Bonus Bets for signing up early between now and Jan.
10, 2024. That means you can receive a total of $300 in Bonus Bets to get you started at FanDuel Sportsbook Vermont.
</item 4>

<item 5>
Vermont Sportsbook Promos and Sports Betting Launch News (Sporting News, 2023-09-26T22:53:28+00:00)
ESPN Bet brings a very familiar brand to sports fans, but a relatively new one when it comes to online sports betting. After acquiring the old Barstool
Sportsbook earlier in the year, ESPN Bet Sportsbook launched the rebranded site in 17 different states in Nov. 2023. The site has already begun taking advantage
of various tie-ins with the renowned sports programming network which has helped in the effort to establish the site in the overall marketplace.

In other states, the ESPN Bet welcome offer has included a 100% deposit match bonus worth up to $1,000 (with a 20x wagering requirement). ESPN Bet has also run
a welcome promo that invites new sign-ups to place a wager of any amount and receive $200 in Bonus Bets split into four $50 Bonus Bets.

Again, we'll keep an eye out for any news regarding ESPN Bet Sportsbook VT.
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet (CBSSports.com, 2023-10-13T21:17:17+00:00)
4. BetMGM Sportsbook: The signup offer at BetMGM caters more to larger players, a $1500 risk free offer which will credit a customer with bonus bets in the
event their first real money wager loses. The boosts and risk free offers on the site cater to smaller players as the limits for those tokens are generally less
than the other sites. With a friendly interface and a rewards program that can be redeemed for bonus bets or MGM Resort credit, this sportsbook is one of the
best in the business for a reason.

5. Caesars Sportsbook: Similar to MGM, the rewards program is what Caesars is most known for in that customers can redeem their wagering credits for use at
Caesars properties. New customers can get up to $1000 in bonus bets if they lose their first bet. The sportsbook offers an extensive betting menu and
specializes in specific boosts related to your home state teams.

6.
</item 8>

<item 9>
Best sports betting promo codes for college basketball: Claim $5,000+ in bonuses from Caesars Sportsbook, BetMGM, Bet365, FanDuel, DraftKings, BetRivers
(Sporting News, 2023-11-29T21:00:04+00:00)
Start by clicking our Caesars Sportsbook promo code link and sign up for a new account. If your first bet loses, Caesars will credit your account with bonus
bets in the amount of your wager up to $1,000.

Click to claim: Caesars Sportsbook promo code NEWS1000 for First Bet up to $1K on Caesars

BetMGM bonus code TSN50 unlocks $1,550 in bonus bets for Blue Devils vs. Razorbacks

Use our exclusive BetMGM bonus code TSN50 to claim a First Bet bonus of up to $1,500, and a $50 bonus bet for customers in select states.

Click our BetMGM bonus code link and register for your new account with promo code TSN50 to unlock this bonus.
</item 9>

<item 10>
Graph query row:
article: Best Golf Betting Sites and Apps - Top Sportsbooks for Golf 2023
published_on: 2023-10-13T21:38:36+00:00
connection: PUBLISHED_BY
outlet: Sporting News
sportsbook: Caesars Sportsbook
</item 10>
=== END ITEM id=0e9d47655a0ee39d ===
