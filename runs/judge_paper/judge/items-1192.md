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

=== ITEM id=8cd0675fa0747f0a ===
Gold sentences and facts:
Sentence 0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 0.0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have off the record chats.
  Fact 0.1: Google will counter this, saying it has provided the court with thousands of chat logs.
  Fact 0.2: Google will counter this, saying it has provided the court with millions of documents.
Sentence 1: Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.
  Fact 1.0: Monday's trial hearing revealed plenty of juicy tidbits.
  Fact 1.1: The tidbits included the $26.3 billion Google spent making itself the default search engine across platforms in 2021.
  Fact 1.2: The tidbits included how Google tried to take its default search engine position further.
  Fact 1.3: The tidbits included how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.4: The hearing revealed other tidbits beyond those listed.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content, their readers and ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“We can tailor our ads to hit exactly the consumers we want,” Moller said. “It’s a perfect marriage of a transaction, and that’s one of the beauties of what
their marketplace offers.” Needham said he feels he would have been more supportive of the case if it were filed a few years ago, pre-pandemic. At that time, he
said, "I would have felt, yes Amazon is a monopoly… But actually after Covid, into 2023, ecommerce has had a lot of big changes.” He added, "The competition is
just not what it was in 2019." Competitors like Shopify and Walmart are increasingly viable alternatives for many categories of sellers, Needham said, not to
mention rapidly growing Chinese e=commerce companies like Temu. As a result, Needham said he’s seen some significant changes from Amazon. Among those is a
greater ability for Amazon sellers to communicate with buyers, offering select customers certain promotions.
</item 1>

<item 2>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Enough that the judge has decided to explicitly tell the jury we’re not seeing all the evidence, because Google employees all the way up to CEO Sundar Pichai
were caught setting those conversations to auto-delete. This issue has come up in the US v. Google antitrust trial in Washington, DC — still ongoing at the
start of Epic’s trial — as well.

And the dirt isn’t all on Google’s side. Epic was caught straight-up admitting it premeditated the whole plan to bypass Google’s payment systems. Since Google
is countersuing over that, it could be a big deal.

When you say “straight-up admitting...”

Here is an email from Epic CEO Tim Sweeney that, I think, speaks for itself:

“Here is our official plan for communicating with Google about bypassing the Google Play Store: SAY NOTHING TILL IT SHIPS” Image via Epic v. Apple

We’ll have a whole cast of intriguing people called as witnesses during the trial who might dish even more.

Who will Epic and Google call as witnesses?
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 7>

<item 8>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 8>

<item 9>
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
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 10>

<item 11>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Or at least it was worse, before they were destroyed.”

The best Google could do in response was to plant its own feeble seed with the jury, too: “Is Epic using the chats to distract me from all the evidence I do
see?”

“It’s true that Google could have automatically saved all chats for all relevant employees, but just because Google didn’t save some chats didn’t mean it
violated antitrust laws,” Pomerantz argued.

Epic’s opening statements seemed to paint a clearer picture for the jury than those from Google. But things got complicated for both parties when the first two
witnesses — Epic Games Store head Steve Allison and Yoga Buddhi CEO Benjamin Simon, who also appeared in the earlier Epic v. Apple trial — took the stand.

Both Epic and Google spent a long, long time on subtle lines of questioning.
</item 11>

<item 12>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 12>

<item 13>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 13>

<item 14>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 14>

<item 15>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 15>

<item 16>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 16>

<item 17>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 17>

<item 18>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We established a market worldwide, excluding China. So any remedies, we would presume, would be worldwide. We also have the European DMA; we have Epic v. Apple
and Epic v. Google cases coming up in Australia, and another one in the UK.

It’s not just Epic anymore, there’s a lot of legislators, there’s a lot of regulators, and there’s other litigation all pushing in the direction of openness.
And we’re going to do absolutely everything we can.

Do you have anything to say to your former partners in litigation, Match Group, that abandoned you at the last minute and may now be regretting it?

Oh, yeah, no. Match has been an awesome partner and a fellow member of the Coalition for App Fairness. I really hope that they got out of their settlement what
they needed to get for their business. Very few companies have the resources that Epic has to fight multinational litigation against the world’s two most
powerful companies.
</item 18>

<item 19>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 19>

<item 20>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
This program, still in pilot testing, is open to all developers who sell apps in the 35 markets where it’s now available, including the U.S., and reduces the
standard commission by 4% for companies who choose to use their own payment processing solution. Spotify and Bumble were the initial testers for the new system,
first introduced in November 2022, and Match, as part of its settlement agreement, will also now take advantage of this option.

Google will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other features and controls to keep Android
users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to build and grow their apps. Google also
helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging from tablets to TVs to autos and more. And
it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.
</item 20>

<item 21>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 21>

<item 22>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
It’s always been in Google’s power to solve this problem. They make several billion dollars a year in unfairly earned profits from imposing this tax, which is
nothing compared to the money they make from search. For all the other benefits they get from Android, Google could solve this problem today if they wanted to.

Do you today believe that Activision Blizzard was really intending to build its own app store?

I don’t know about Supercell, but we know from the documents in the case that Riot was planning to distribute League of Legends directly through their website,
on mobile, exactly as they do on PC. And that’s what they planned to do until Google paid them off to not do that. Google’s payoff at the minimum dissuaded Riot
from distributing off Google Play.

I think they’re the strongest example of the three, but I’m curious about Activision.

Activision was building a store! We know they had a large team of dozens of engineers building a mobile app store to launch on Android.
</item 22>

<item 23>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
When should we expect a jury verdict?

Judge Donato told jurors that we’re looking at roughly five weeks of trial starting November 6th and extending through early December. The schedule suggests the
court will take Thanksgiving off, finish looking at all the evidence by December 4th, take a couple more days off before closing arguments, and then let jurors
come to a decision.

“I fully expect we’ll have plenty of time to reach a verdict well before the December holiday period,” said Judge Donato.

Okay, I’m sold — how can I watch the trial unfold?
</item 23>

<item 24>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
showed that Epic knew it was breaking bad at the time it did the deed.

But Epic mentioned Project Liberty in its own opening statement — so, by that point, it had already been an hour since it admitted it intentionally broke
Google’s rules. “Epic decided to stand up because that’s what you do to a bully,” Bornstein told the jury.

“All we know is whatever is in the destroyed chats, as bad as the documents are, is worse.”

And it’s possible no examination will be able to take the stink off one of Google’s ugliest moves: the one where Google employees up to and including CEO Sundar
Pichai were caught setting sensitive chats to auto-delete to keep them out of a court’s hands. The court has already decided Google should be sanctioned in some
way for making potential evidence disappear, and Bornstein used it to plant persistent seeds of doubt in the minds of jury members. “All we know is whatever is
in the destroyed chats, as bad as the documents are, is worse.
</item 24>

<item 25>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 25>

<item 26>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
That was our proposal.

Oh, I think I read that one in court. I didn’t realize that was the same thing. It’s so far off from what I heard [about a “special billing exception”] that I
didn’t recognize it being that. Does winning the verdict in this Google case help your appeal with the Apple case in any way?

There’s no linkage between the cases and law, so it would just come down to whether the court is in any way following current events on this topic. But there’s
no legal connection between the two. It really comes down to the decisions of the humans involved: the justices and the support teams involved when considering
the appeal.

Google will appeal this; they’ve told us they’re going to challenge the verdict. Will Epic rely on this ruling in the meantime?

We’re not going to wait. We’re going to do absolutely everything we can as quickly as we can to start changing the world. We not only have this verdict here in
the United States, it is a worldwide verdict, right?
</item 26>

<item 27>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 27>

<item 28>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 28>

<item 29>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 29>

<item 30>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 30>

<item 31>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 31>

<item 32>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 32>

<item 33>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 33>

<item 34>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
[Editor’s note: Every journalist in the courtroom agreed the jury was attentive; most were also surprised by the rapid verdict.]

They got it, they got it quickly, and they were able to pull apart what was going on and contrast it with the fiction of the story that Google was trying to
tell.

What was something that surprised you in the courtroom, something you heard come to light that you hadn’t heard before?

I’d suspected a lot of the practices that Google had, you know, since 2018 or so when we first started this, to such an extent that some folks would
occasionally call me a conspiracy theorist. It was really, really interesting to see that my understandings of what Google was doing behind the scenes were
actually true — you’re leaking our conversations to reporters to get negative stories written about us; you’re paying other developers off to convince them not
to launch their own stores; they were going around and paying carriers and OEMs secretly not to carry competing stores.
</item 34>

<item 35>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 35>

<item 36>
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
</item 36>

<item 37>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 37>

<item 38>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.
</item 38>

<item 39>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.
</item 39>

<item 40>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 40>

<item 41>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 41>

<item 42>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 42>

<item 43>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray reassured me that I was not crazy. Google results today do feel different from how they felt just five or six years ago for two major reasons. The first was
Google’s response to the disinformation panic around the 2016 election, which involved questioning the notion that the most reliable information could be chosen
by a form of popularity, meaning how many links a site received from other sites. As a result, the algorithm seemed to change its approach to links, especially
when it came to news and sites offering legal, financial, or health advice, and instead paid more attention to what Google came to call E-E-A-T: experience,
expertise, authoritativeness, and trustworthiness.

“E-E-A-T has had a pretty big impact on what types of results you see,” Ray told me. She’s done extensive (and fascinating) research around how certain sites
have fared under these new guidelines: Urban Dictionary, down! Mayo Clinic, up! Some people consider EEAT part of what’s making results better than ever.
</item 43>

<item 44>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 44>

<item 45>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So this trial has been four years in the making. I look back at a September 2019 email that was in discovery about a plan to draw Google into a legal battle
over antitrust. Can you describe what it felt like, after four years, to hear the jury find for Epic?

Well, it’s a great day for all developers to see that the Sherman Antitrust Act works in the new era of tech monopolies; we’ve not had a major antitrust verdict
against a tech company that meant change and benefits for everybody since the 1990s, with the US v. Microsoft. Back in the early days of the internet. So this
is an awesome thing and it’s much needed by the industry which is being strangled by a few gatekeepers imposing insane amounts of control and extracting huge
taxes, which not only raise prices for consumers but also make a lot of kinds of products just unviable.

I understand the significance, but... you were there in person.
</item 45>

<item 46>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 46>

<item 47>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 47>

<item 48>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 48>

<item 49>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I understand the significance, but... you were there in person. You had a smile on your face, you shook the Google attorney’s hand, you clapped Bornstein on the
back. How did you feel in that moment?

Well, it was a great relief. The conventional wisdom that attorneys tell you is that when there’s a rapid jury verdict, it’s typically not good for the
plaintiffs making a complicated case, and so there was some trepidation going on — but it was awesome to see.

Something that we’d suspected all along was the jury was really following the case carefully. They weren’t snoozing off as you might expect with the complexity
of these documents and things — everybody was paying attention, they’re looking at the question asker and the witness and taking notes and reading documents. It
was just awesome to see that the justice system works, even with the most complex type of tech antitrust case that you can possibly throw at a jury.
</item 49>

<item 50>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 50>

<item 51>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 51>

<item 52>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 52>

<item 53>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 53>

<item 54>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 54>

<item 55>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Looking at this from a less academic perspective, it’s worth noting that Google charges up to ten times more per transaction than you pay with PayPal or a
credit card, which does seem like a lot! And though Epic can’t argue it in court, I’ll point out that Apple wasn’t able to explicitly justify a 30 percent fee
to that case’s judge.

On the other hand, it does kinda seem like Epic wants something for nothing! As far as I can tell, Epic hasn’t stated what kind of fee would actually be
reasonable for Google to charge in exchange for placement in Google’s store — instead, Epic CEO Tim Sweeney seemingly suggests there shouldn’t be a Google fee
at all if a developer wants to use its own payment system.

How did Epic’s argument fare against Apple?

Well… both sides lost! But Epic arguably lost more.
</item 55>

<item 56>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 56>

<item 57>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 57>

<item 58>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
As an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one developer that wanted to go the
route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020, worth approximately $360 million,
to keep the developer on Google Play, Epic will argue.

Google, however, will characterize this program as a way to incentivize developers to launch their apps on Android at the same time as they launch on iOS and
plans to dispute Epic’s claims that it prevented developers from opening their own app stores.

Android allows sideloading

Google will make note that, unlike Apple, it also allows apps to be sideloaded on Android devices — something Epic takes advantage of today. It will note this
process is streamlined and helps keep users safe.
</item 58>

<item 59>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 59>

<item 60>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
The lawyers can see them, but I saw almost all the key elements in this case at the same time the jury saw them. Two things stand out big time. Google’s Project
Hug was an astonishingly corrupt effort at a massive scale — the same corrupt deal structure they tried to do with Epic when they wanted to pay us off to launch
Fortnite on Google Play and not launch our own store.

As soon as we thwarted their effort, they went around to 27 different developers and offered each one a payoff to undermine any effort we had to get their games
onto our store exclusively. Activision and Riot and Supercell had direct distribution plans that they were planning on; Google paid them not to pursue those
plans. Just direct blatant violations of anti-competition law, it’s crazy a company of Google’s scale would do that.

“If you’re a smaller developer than Spotify, you get screwed.”

The other astonishing one was the Spotify deal. Spotify is the one company that had comparable negotiating power to Fortnite.
</item 60>

<item 61>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 61>

<item 62>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 62>

<item 63>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 63>

<item 64>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
It’s difficult to extrapolate from the 2015 and 2020 numbers to today’s (which, as we have seen from the evidence presented here, will almost certainly not be
the whole story), but Bonta noted that the large figures are presented for impact, not as legal justification.

“The basic premise remains that their social media platforms are used by millions of children under 13. Whether it’s 30 percent, or 20 or 10 percent… any child,
it’s illegal,” he said. “If they were doing it at any time, it violated the law at that time. And we are not confident that they have changed their ways.”

An internal presentation called “2017 Teens Strategic Focus” appears to specifically target kids under 13, noting that children use tablets as early as 3 or 4,
and “Social identity is an Unmet need Ages 5-11.”
</item 64>

<item 65>
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
</item 65>

<item 66>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 66>

<item 67>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Who will Epic and Google call as witnesses?

We are going straight to the top of the org chart, folks. Google CEO Sundar Pichai and Epic CEO Tim Sweeney are absolutely expected to attend. Android
co-founder Andy Rubin might be deposed, and Google says we’ll hear from current Android (and Chrome, and Chrome OS) boss Hiroshi Lockheimer. Google also tells
us representatives from Apple, Netflix, Motorola, and AT&T should testify as well.

How much hypocrisy can we expect in the courtroom?

I’d say quite a bit! For example:

Epic says it’s not in this for the money, it’s only suing to end Google’s monopoly — but Epic would surely profit if it doesn’t have to pay Google’s cut.

It looks like Epic’s going to argue that you’re locked into Android after buying an Android phone, but I can personally attest that switching back and forth
between the platforms is easier than ever.
</item 67>

<item 68>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 68>

<item 69>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Not Sonos fast there: Audio company Sonos scored a big $32.5 million win against Google a while back. Now it transpires that the company’s patents were deemed
unenforceable and invalid. Whoops. A federal judge threw out the $32.5 million win this week.

When it clicks, it really clicks: We took a deep look at Pixel 8, and our team discovered that it delivers solid performance and design upgrades. The camera got
a particularly enthusiastic shout-out, with Darrell declaring that Google’s Pixel 8 Pro camera is the new mobile photography champ, and Brian waxing lyrical
about the phone’s picture-snapper in “The camera’s still the thing.”

Who needs computers anyway?: It seems like all our mobile devices may be starting to cannibalize sales of personal computers — Ron reports that PC shipments
decline slows in Q3 2023 and that Apple’s sales plunge over 23%.

Let’s get together

Reddit’s API-powered chaos continues, but it appears that things are starting to resolve a little.
</item 69>

<item 70>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The end result of these tactics, the FTC argues, is elevated prices across the web. The company steadily hikes the fees it charges sellers and prevents them
from discounting on other sites, so sellers often inflate their prices off of Amazon, creating an “artificial price floor everywhere,” according to the
complaint. The FTC is seeking to hold Amazon liable for allegedly violating anti-monopoly law, though it has not yet outlined the specific remedies it believes
would best resolve its concerns. In antitrust cases, remedies are often determined only after a court finds the defendant liable. In a blog post, Amazon general
counsel David Zapolsky said third-party sellers set their own prices on the marketplace. The company also invests in tools to help sellers offer “competitive
prices,” he said. “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.
</item 70>

<item 71>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 71>

<item 72>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I don’t know if that was on behalf of you and fellow developers. Can you tell us what that email was actually about?

Google didn’t leak the email; they leaked their ridiculously biased summary of it to Abner Li at 9to5Google. Don Harrison testified at trial that he didn’t
think Google leaked to the press. And then he was presented with a document including Google’s press team and Sameer Samat, head of all Android, basically
summarizing the articles that appeared as a result of Google’s leaking of our plans to the press.

So I was wondering if you could tell me what the email actually said.

Oh, it’s in evidence! This was the one I mentioned! Right before Christmas-ish 2019, we were planning to launch the new Marvel season with a bunch of Star Wars
content and its awesome lineup. I asked Google’s executives to let Fortnite come to the Google Play Store using our own payment method and to open up, you know,
using their own payment methods to all developers, not just Epic.
</item 72>

<item 73>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 73>

<item 74>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Just like any store owner who wouldn’t want to promote a bad deal to their customers, we don’t highlight or promote offers that are not competitively priced.”
Zapolsky argued the FTC’s lawsuit could force it to stop highlighting low prices, “a perverse result that would be directly opposed to the goals of antitrust
law.”

“Long overdue” lawsuit

On Amazon’s own forum for merchants, called Seller Central, several users cheered on the FTC and said they hoped it would result in changes to the company’s
business practices. Amazon’s tense relationship with merchants has been well-chronicled over the years, with sellers expressing a range of grievances over
issues like rising fees, an arcane suspensions process, and heightened competition on the marketplace from all sides, including the e-commerce giant. “I think
it’s great, Amazon deserves it,” one person commented, adding, “More should be coming on the way.”
</item 74>

<item 75>
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
</item 75>

<item 76>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 76>

<item 77>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”
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
</item 79>

<item 80>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 80>

<item 81>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 81>

<item 82>
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
</item 82>

<item 83>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 83>

<item 84>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He speculated that the company didn’t want the public to think too hard about how search works because that awareness “encourages regulators and makes people
realize, ‘Oh, there’s a lot of money here.’”

There has always been advertising and polemics from cranks, scammers, and liars. But now we see this stuff surfacing alongside truth, and we can’t tell the
difference. We move through our lives with a greater sense of distrust and fear and insecurity. At the end of the day, it’s Google’s world, and the SEOs are
only living in it.

And as much as I might hate the way the SEOs who don’t follow Google’s rules have altered my online experience, the reality is that most people running a
company will break whatever rules they are able to get away with breaking.
</item 84>

<item 85>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Could Google actually lose?

When I walked into the courtroom on Monday morning, it seemed impossible. If Epic couldn’t prove Apple’s walled iOS garden is a monopoly, how could the
comparatively open Google do worse against the windmill-tilting Fortnite developer?

But now that both sides have made their opening arguments to a jury, I’m not quite as sure. Because while Google spent most of its first day attempting to
explain complicated ins and outs of business, Epic was able to paint a black-and-white picture of good and evil with itself as the clear underdog.

Epic lead attorney Gary Bornstein was tasked with making the case that Android functions as an unlawful monopoly. He did so by basically calling Google a bully
and a cheat that “bribes” or “blocks” any attempt to compete with Android’s Google Play store. The result? A status quo where the vast, vast majority of Android
app installs are from Google Play, with only a tiny sliver attributable to the Galaxy Store that comes preinstalled on every Samsung phone.
</item 85>

<item 86>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 86>

<item 87>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
No, we’ve always turned down special deals just for Epic. We’ve always fought on the principal that all developers should be, you know, given the same
opportunities. One of the documents in evidence was a 2019 email I sent — right before Fortnite launched the Marvel season with all this amazing new content and
the Star Wars event featuring J.J. Abrams. Right before that, I sent an email to all the Google senior executives saying that we wanted to bring Fortnite to the
Google Play Store in time for that event, and we wanted them to allow us and all other developers to process certain payments and keep them all.

That was our proposal to Google in 2019. If Google had said yes to that, that would have been awesome for all developers — the Android ecosystem would have
become much, much stronger, and Google would be in a much better position in the smartphone industry than they are today. We would’ve never had a dispute
because the problem would have been solved.

It’s always been in Google’s power to solve this problem.
</item 87>

<item 88>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 88>

<item 89>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Also, that one’s not over till the Supreme Court weighs in or declines to take a look.

2) Google can’t tell jurors that Apple won its case, or that other plaintiffs settled. The judge in this case says so!

3) Oh, and by the way, this is a jury trial.

Wait, why does that matter?

Epic and Google have to convince a jury, not the judge, which is totally different from how the Apple case played out. (That one was a “bench trial.”)

Maybe all the evidence of tricky deals inside Google might sway a jury against the company? Maybe Google scaremongering that sideloaded apps equal gaping
security holes will sway a jury against Epic instead? Who knows!

(If you’re a Epic juror reading this — stop! Judge Donato explicitly said you’re in a “news-free bubble” through mid-December, folks.)

Didn’t other parties sue Google too?

They did! And then, they all settled.
</item 89>

<item 90>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 90>

<item 91>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thank you for being here with us. It’s been a very engaging trial to watch. I’ve been there every day of the trial, and you’ve been there every day save one. So
my first question is why did you personally attend this trial every day save one — and what the heck happened on that one day?

Yeah, Epic is asking a lot of the court system and the jury here, spending four weeks on a major antitrust trial full of complicated facts and evidence. It
wouldn’t be right to start something like this and not show up. So I had to do that. And, you know, Phil Schiller sat throughout the entire Epic v. Apple trial,
as did I, so I think it’s just necessary to show respect for the legal process.

What happened on that one day you weren’t here?

Oh — sorry I can’t share, but there were some... non-Epic issues I had to deal with.

So this trial has been four years in the making.
</item 91>

<item 92>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 92>

<item 93>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
I did like Pomerantz’s commonsense argument that Google can’t possibly have a monopoly on Android app stores when “every single Samsung phone comes with two app
stores right on the homescreen,” which continued:

When they show these charts that show all these downloads from Play and not from the Galaxy Store, that’s what the Samsung phone users are choosing. They’re
touching Play. Nothing’s keeping them from touching the Galaxy Store; it’s just what works for them.

I called Google “comparatively open” earlier, and that openness will likely be heavily debated in the weeks to come. Epic promised to “show that Google has
closed off each and every other option” to the Play Store during this trial. But Google points to the simple fact that it allows alternate app channels at all —
something Android rival iOS doesn’t.

Pomerantz boasted that over a billion people have gone through the process Epic portrays as needlessly onerous to get apps outside the Play Store.
</item 93>

<item 94>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Google spent its 45 minutes attempting to explain that its dominance over the Android app market isn’t anything nefarious but simply the natural outcome of
Google fiercely competing with the iPhone and its iOS App Store, where Google would like the court to believe that competition truly lies.

If Google can convince the jury of that, it could be a winning argument in the case — because obviously, Google doesn’t have a monopoly on app stores or phones
in general. “You cannot separate the quality of a phone from the quality of the apps in its app store, and that means Google and Apple compete against each
other,” began Google lead attorney Glenn Pomerantz.

But Google wound up spending much of its opening statement attempting to explain away its seemingly bad behavior as normal business practices and didn’t always
succeed out of the gate.
</item 94>

<item 95>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 95>

<item 96>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
she asked. “Or is this Google behaving differently than it used to?”

Sullivan had tried to convince me that Google was not behaving differently and, in fact, had not changed its search criteria in any major way for the past 20
years. Google wanted you to make good websites, and that was that. Everyone who tried to rank higher by messing with the algorithm would be blocked. Sullivan
even insisted that what these rule-breakers did should not be called SEO: he deemed it all “spam.” What is spam? “Spam is stuff that search engines don’t like.”

But the line between strategies that violate Google’s terms of service and strategies that don’t has always been blurry and inconsistently enforced. “I’ve never
seen this much tension in the industry in terms of, like, what Google says to do and what people are doing and getting away with,” Ray told me. “If you’re gonna
tell us that this stuff doesn’t work, make it stop working!”

Ray seemed like the most reasonable person I had spoken to so far.
</item 96>

<item 97>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 97>

<item 98>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 98>

<item 99>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
He hoped to make Google and Apple’s services so connected as to be inseparable.

The way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention all that juicy data —
and Apple would get more revenue, as a result of the lucrative search agreement the two signed.

For whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.

Google’s deleted chat logs

During Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In February, the
DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually changed the
setting.

Pichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according to CNBC.
</item 99>

<item 100>
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
</item 100>
=== END ITEM id=8cd0675fa0747f0a ===
