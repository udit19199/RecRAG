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

This file has 4 items.

=== ITEM id=c79851dcd5810261 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 0.2: The case argues that Google siphons off news publishers' readers.
  Fact 0.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 0.4: The case argues that Google does this through anticompetitive means.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed an article for not meeting its standards.
  Fact 1.1: The article was shared widely on social media.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The degree to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.

Retrieved items:
<item 1>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 1>

<item 2>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 3>

<item 4>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
This law created very specific restrictions around young folks online, limiting data collection and requiring things like parental consent for various actions,
but a lot of tech companies seem to consider it more suggestion than requirement.

You know it is bad news for the company when they request pages and pages of redactions:

This recently happened with Amazon as well, and it turned out they were trying to hide the existence of a price-hiking algorithm that skimmed billions from
consumers. But it’s much worse when you’re redacting COPPA complaints.

“We’re very bullish and confident in our COPPA allegations. Meta is knowingly taking steps that harm children, and lying about it,” AG Bonta told TechCrunch in
an interview. “In the unredacted complaint we see that Meta knows that its social media platforms are used by millions of kids under 13, and they unlawfully
collect their personal info. It shows that common practice where Meta says one thing in its public-facing comments to Congress and other regulators, while
internally it says something else.”
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 5>

<item 6>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 6>

<item 7>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 7>

<item 8>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 8>

<item 9>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 10>

<item 11>
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
</item 11>

<item 12>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 12>

<item 13>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
[Editor’s note: Every journalist in the courtroom agreed the jury was attentive; most were also surprised by the rapid verdict.]

They got it, they got it quickly, and they were able to pull apart what was going on and contrast it with the fiction of the story that Google was trying to
tell.

What was something that surprised you in the courtroom, something you heard come to light that you hadn’t heard before?

I’d suspected a lot of the practices that Google had, you know, since 2018 or so when we first started this, to such an extent that some folks would
occasionally call me a conspiracy theorist. It was really, really interesting to see that my understandings of what Google was doing behind the scenes were
actually true — you’re leaking our conversations to reporters to get negative stories written about us; you’re paying other developers off to convince them not
to launch their own stores; they were going around and paying carriers and OEMs secretly not to carry competing stores.
</item 13>

<item 14>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
During his time at Google, Cutts regularly received death threats and hate mail. When SEOs would send, say, a fruit plate or a brownie cake addressed to him at
Google’s offices, he told me, “We’d take it down to the kitchen with a note warning: possibly poisoned.”

After Cutts left, Google replaced him with a handful of people, none of whom could quite fill his shoes: “Those personalities sometimes were standoffish,”
Forrester told me. “Some of them were superior. Some of them were a bit too wallflower.”

One of the people Google brought in was Danny Sullivan, a former journalist who started Search Engine Land, the industry publication where Schwartz works, back
in the 2000s.
</item 14>

<item 15>
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
</item 15>

<item 16>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 16>

<item 17>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Although many of us may have rosy memories of how magical and cool Google seemed in the early days, most SEOs consider the years between 2003 and 2011 to be the
boom times, when you could still get a fake corporate website listed above the real corporate website, and you could mess with the search results for a major
political figure such that something sexual or racist would come up first.

Google is harder to game now — it’s true. But the sheer volume of SEO bait being produced is so massive and so complex that Google is overwhelmed. “It’s
exponentially worse,” Ray said. “People can mass auto-generate content with AI and other tools,” she went on, and “in many cases, Google’s algorithms take a
minute to catch onto it.”

The future that Babin had cackled about at the alligator party was already here. We humans and our pedestrian questions were getting caught up in a war of
robots fighting robots, of Google’s algorithms trying to find and stop the AI-enabled sites programmed by SEOs from infecting our internet experience.
</item 17>

<item 18>
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
</item 18>

<item 19>
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
</item 19>

<item 20>
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
</item 20>

<item 21>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 21>

<item 22>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 22>

<item 23>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He speculated that the company didn’t want the public to think too hard about how search works because that awareness “encourages regulators and makes people
realize, ‘Oh, there’s a lot of money here.’”

There has always been advertising and polemics from cranks, scammers, and liars. But now we see this stuff surfacing alongside truth, and we can’t tell the
difference. We move through our lives with a greater sense of distrust and fear and insecurity. At the end of the day, it’s Google’s world, and the SEOs are
only living in it.

And as much as I might hate the way the SEOs who don’t follow Google’s rules have altered my online experience, the reality is that most people running a
company will break whatever rules they are able to get away with breaking.
</item 23>

<item 24>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 24>

<item 25>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
As an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one developer that wanted to go the
route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020, worth approximately $360 million,
to keep the developer on Google Play, Epic will argue.

Google, however, will characterize this program as a way to incentivize developers to launch their apps on Android at the same time as they launch on iOS and
plans to dispute Epic’s claims that it prevented developers from opening their own app stores.

Android allows sideloading

Google will make note that, unlike Apple, it also allows apps to be sideloaded on Android devices — something Epic takes advantage of today. It will note this
process is streamlined and helps keep users safe.
</item 25>

<item 26>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
It was easy: buy links that led to your site and watch your ranking in Google’s results rise.

I came to understand that, since the dawn of the internet, there have been people attempting to manipulate search and then people decrying those manipulations
as the end of search’s ability to be useful. It works in cycles. People doing SEO find loopholes in the algorithm; critics complain about search results; search
engines innovate and close the loopholes. Rinse, repeat.

Before our current moment of widespread disillusionment with online information, the rise of SEO had reached a breaking point multiple times. In 2003, as Google
approached the deadline to disclose pertinent business information leading up to its IPO, the company quietly released an update cracking down. By 2011, SEO was
once again oppressively pervasive. TechCrunch published a story called “Why We Desperately Need a New (and Better) Google,” which argued that “Google has become
a jungle: a tropical paradise for spammers and marketers.”
</item 26>

<item 27>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 27>

<item 28>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Some people consider EEAT part of what’s making results better than ever. Others see it as a form of censorship, disproportionately affecting right-wing
perspectives. Not every search query takes EEAT into account; Google has described heightened concern over sites that could impact safety, happiness, and the
ability to be an informed citizen. But the point that really hit me was that for certain kinds of information, Google had undone one of the fundamental elements
of what had made its results so appealing from the start. Now, instead of wild-west crowdsourcing, search was often reinforcing institutional authority.

You can’t just be the most powerful observer in the world for two decades and not deeply warp what you are looking at

This felt complicated at best. When it comes to health and wellness, for example, quackery is often in the eye of the beholder. Everyone knows someone who has
struggled with the limits of Western medicine. So much of the original draw of the internet was the opportunity for outlier voices to be heard alongside
established experts and elites.
</item 28>

<item 29>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 29>

<item 30>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 30>

<item 31>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray seemed like the most reasonable person I had spoken to so far. Sure, she called herself a “thought leader,” and yes, sure, she had changed her last name to
improve her personal branding by more closely associating herself with her grandmother’s uncle, the artist Man Ray. Maybe some people would say that’s the kind
of absurd behavior that merits being attacked by an alligator, but I was beginning to come down on the side of the SEOs, who seemed to have a lot less agency
than I’d first imagined.

Google had started with a noble cause: trying to make the internet easier to navigate at scale. The company did accomplish that goal, but in doing so, it
inadvertently and profoundly changed how the internet looked. The problem lay in Google trying to be an objective and neutral arbiter of an information
landscape that was meant to pretend it did not exist. You cannot design a free, automated system to help people find information without some people trying to
game that system.
</item 31>

<item 32>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Google’s software is constantly crawling the web and updating its central index, which itself is more than 100 million gigabytes in size. When you enter a
query, Search uses a few quick processes to decide the meaning of your question, including what kind of information you might be after. Then, beyond just
looking for sites with the words you typed in, the system uses machine learning to rank results according to what it thinks you’re looking for, prioritising
high quality and easy to use sites. If you allow it, Google does bring in some information it knows about you to further refine results, including your location
and web history. If you’re looking to turn this off, it’s in your Google account settings called “Web & App Activity”. Google says it does not infer personal
information like race or religion to shape results. Loading Google works with independent “search quality raters” to gauge the effectiveness of its processes,
and says it ran almost 900,000 quality tests in 2022.
</item 32>

<item 33>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
showed that Epic knew it was breaking bad at the time it did the deed.

But Epic mentioned Project Liberty in its own opening statement — so, by that point, it had already been an hour since it admitted it intentionally broke
Google’s rules. “Epic decided to stand up because that’s what you do to a bully,” Bornstein told the jury.

“All we know is whatever is in the destroyed chats, as bad as the documents are, is worse.”

And it’s possible no examination will be able to take the stink off one of Google’s ugliest moves: the one where Google employees up to and including CEO Sundar
Pichai were caught setting sensitive chats to auto-delete to keep them out of a court’s hands. The court has already decided Google should be sanctioned in some
way for making potential evidence disappear, and Bornstein used it to plant persistent seeds of doubt in the minds of jury members. “All we know is whatever is
in the destroyed chats, as bad as the documents are, is worse.
</item 33>

<item 34>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
It’s difficult to extrapolate from the 2015 and 2020 numbers to today’s (which, as we have seen from the evidence presented here, will almost certainly not be
the whole story), but Bonta noted that the large figures are presented for impact, not as legal justification.

“The basic premise remains that their social media platforms are used by millions of children under 13. Whether it’s 30 percent, or 20 or 10 percent… any child,
it’s illegal,” he said. “If they were doing it at any time, it violated the law at that time. And we are not confident that they have changed their ways.”

An internal presentation called “2017 Teens Strategic Focus” appears to specifically target kids under 13, noting that children use tablets as early as 3 or 4,
and “Social identity is an Unmet need Ages 5-11.”
</item 34>

<item 35>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 35>

<item 36>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
But the result left it unclear how cookie paywalls might be operated in a way that’s GDPR compliant and the privacy rights group vowed to fight the decision in
court. “The final decision on ‘pay or okay’ may be made by the European Court of Justice (CJEU) in the long run,” noyb predicted at the time.

Meta is likely banking on another multi year round of GDPR complaints, legal challenges and — finally — a referral to the CJEU, followed by another long wait
before a ruling gets handed down, buying it several more years to run with its new legal basis fix and keep feeding its profits by doing what it likes with
Europeans’ data.

But the consumer protection challenge could complicate its usual playbook.

The CPC has brought more coordinated action on consumer protection concerns in the EU in recent years, bringing multiple consumer groups together to tackle
common concerns — helped by one or more national consumer protection authority which gets appointed to drive the effort.
</item 36>

<item 37>
European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ (TechCrunch, 2023-11-30T05:00:57+00:00)
Some DPAs appear to have been sympathetic to local newspapers’ use of cookie walls, seeing it as a way to support the production of journalism. However when it
comes to Meta, that argument evaporates as it’s definitely not in the journalism business. Moreover the adtech giant doesn’t even need to produce content to
pump around its social networks; it gets all that filler for free from the self-same users it’s now demanding pay a fee if they want to use its services without
being tracked and profiled for behavioral advertising. Which, well, makes Meta’s ‘pay or okay’ model feel like even more of a rip off.

Back in April, a decision by Austria’s DPA on a noyb complaint about cookie paywalls said users must have the ability to say yes or no to specific data
operations — meaning blanket consent is not an option.
</item 37>

<item 38>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 38>

<item 39>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
she asked. “Or is this Google behaving differently than it used to?”

Sullivan had tried to convince me that Google was not behaving differently and, in fact, had not changed its search criteria in any major way for the past 20
years. Google wanted you to make good websites, and that was that. Everyone who tried to rank higher by messing with the algorithm would be blocked. Sullivan
even insisted that what these rule-breakers did should not be called SEO: he deemed it all “spam.” What is spam? “Spam is stuff that search engines don’t like.”

But the line between strategies that violate Google’s terms of service and strategies that don’t has always been blurry and inconsistently enforced. “I’ve never
seen this much tension in the industry in terms of, like, what Google says to do and what people are doing and getting away with,” Ray told me. “If you’re gonna
tell us that this stuff doesn’t work, make it stop working!”

Ray seemed like the most reasonable person I had spoken to so far.
</item 39>

<item 40>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Because they haven’t, he insisted. If anything, search results have gotten a lot better over time. Anyone who thought search quality was worse needed to take a
hard look in the mirror.

“We have an entire generation that grew up expecting the search box to do the work for them,” he said. “We might do a better job of matching for a bulk of
people, but for people who are super sensitive, when they have that fail moment, now it becomes, ‘All my searches aren’t good.’”

The problem was not Google. The problem was not SEOs. The problem was kids these days.

Of course Sullivan would say this, though. He works for Google. I felt like I began to understand why many SEOs had told me that Cutts’ departure had marked a
major turning point in the history of the internet, emblematic of Google’s transition from idealistic startup to one of the most valuable and powerful companies
to ever exist.
</item 40>

<item 41>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
The lawyers can see them, but I saw almost all the key elements in this case at the same time the jury saw them. Two things stand out big time. Google’s Project
Hug was an astonishingly corrupt effort at a massive scale — the same corrupt deal structure they tried to do with Epic when they wanted to pay us off to launch
Fortnite on Google Play and not launch our own store.

As soon as we thwarted their effort, they went around to 27 different developers and offered each one a payoff to undermine any effort we had to get their games
onto our store exclusively. Activision and Riot and Supercell had direct distribution plans that they were planning on; Google paid them not to pursue those
plans. Just direct blatant violations of anti-competition law, it’s crazy a company of Google’s scale would do that.

“If you’re a smaller developer than Spotify, you get screwed.”

The other astonishing one was the Spotify deal. Spotify is the one company that had comparable negotiating power to Fortnite.
</item 41>

<item 42>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
They choose to do things that maximize the frequency of use of that platform by children, and the duration of that use. They could end all this today if they
wanted, they could easily keep those under 13 from accessing their platform. But they’re not.”

You can read the mostly unredacted complaint here.

(This story has been updated with a comment from Meta.)
</item 42>

<item 43>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Were all of us wrong? Or engulfed in some kind of Baader–Meinhof frequency bias delusion? Had I been researching a nonexistent problem? Were Google results
actually amazing? Truly, I had lost the plot. Was the premise of this piece completely off? Was I the asshole who deserved to be attacked by an alligator?

I began to worry all the people who were mad about search results were upset about something that had nothing to do with metrics and everything to do with
feelings and ~vibes~ and a universal, non-Google-specific resentment and rage about how the internet has made our lives so much worse in so many ways, dividing
us and deceiving us and provoking us and making us sadder and lonelier. Decades of American optimism about the wonderful potential of technology, from the Moon
landing to personal computers to the iPhone, had finally, in the last few years, broken down into comprehensive chagrin at the petty, pathetic, and violent
world enabled by our devices. Was all that really Google’s fault?
</item 43>

<item 44>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 44>

<item 45>
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
</item 45>

<item 46>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 46>

<item 47>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The EU’s voluntary anti-disinformation Code has 44 signatories in all — covering not just major social media and search platforms such as the aforementioned
giants but entities from across the ad industry and civil society organizations involved in fact-checking.

Google

On generative AI, Google’s report discusses “recent progress in large-scale AI models” which it suggests has “sparked additional discussion about the social
impacts of AI and raised concerns on topics such as misinformation”. The tech giant is an early adopter of generative AI in search — via its Bard chatbot.

“Google is committed to developing technology responsibly and has published AI Principles to guide our work, including application areas we will not pursue,” it
writes in summary on the topic, adding: “We have also established a governance team to put them into action by conducting ethical reviews of new systems,
avoiding bias and incorporating privacy, security and safety.

“Google Search has published guidance on AI-generated content, outlining its approach to maintaining a high standard of information quality and the overall
helpfulness of content on Search.
</item 47>

<item 48>
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
</item 48>

<item 49>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 49>

<item 50>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Cutts, as an individual, seemed to be doing his best within an expanding corporate behemoth to remain approachable. “One thing I learned early on was that even
when someone was shouting at you, there’s a kernel of something you needed to hear in the other person and listen to and respect and integrate and incorporate,”
he told me. Most SEOs told me they appreciated his efforts. When Google released the 2011 Panda update that devastated a generation of SEO businesses, Cutts
openly recognized the impossible task of achieving the kind of epistemological neutrality that Google’s founders had initially promised, telling Wired at the
time, “[T]he only way to be neutral is either to randomize the links or to do it alphabetically.”

Still, some blamed him personally for “killing” companies that had relied on the previous iteration of the algorithm. During his time at Google, Cutts regularly
received death threats and hate mail.
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
</item 52>

<item 53>
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
</item 53>

<item 54>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 54>

<item 55>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 55>

<item 56>
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
</item 56>

<item 57>
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
</item 57>

<item 58>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 58>

<item 59>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 59>

<item 60>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 60>

<item 61>
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
</item 61>

<item 62>
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
</item 62>

<item 63>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 63>

<item 64>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Over the phone, Cutts came off as humble and thoughtful, acknowledging the nuances and challenges of the search engine business, while Sullivan sounded like an
impatient corporate stooge, trying to gaslight me into believing the sky was red.

But here’s the part where I started to feel the way I’ve felt so often in recent years, like I was losing my grip on reality: Sullivan was not the only person
who tried to tell me that search results have improved significantly. Out of the dozen-plus SEOs that I spoke with at length, nearly every single one insisted
that search results are way better than they used to be. And except for Sullivan, these were not people with an incentive to praise Google. If anything, these
were folks who lamented how much harder it had become for them to take advantage of Google. Today, they told me, search results are just objectively more
accurate. More useful. More difficult to manipulate.

This was not what I had been noticing, and this was certainly not what I had been hearing from friends and journalists and friends who are journalists.
</item 64>

<item 65>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 65>

<item 66>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
In the beginning, the internet was seen as an improvement on the spirit of the public library. Here was an opportunity to transcend the gatekeepers controlling
who could publish a book, allowing mankind to fully connect and share knowledge. Instead, we have ended up in a situation arguably worse than before, where
nearly all online information runs through a single company, which assumes a veneer of civic utility, of impassive authority, when it is very much not a neutral
entity.

“There were so many true believers at Google in the early days,” Cutts told me. “As companies get big, it gets harder to get things done. Inevitably, people
start to think about profit or quarterly numbers.” He claimed that, at least while he was there, search quality always came before financial goals, but he
believes that the public underestimates how Google is shaping what they see, saying, “I deeply, deeply, deeply believe search engines are newspaper-like
entities, making editorial decisions.”
</item 66>

<item 67>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
This program, still in pilot testing, is open to all developers who sell apps in the 35 markets where it’s now available, including the U.S., and reduces the
standard commission by 4% for companies who choose to use their own payment processing solution. Spotify and Bumble were the initial testers for the new system,
first introduced in November 2022, and Match, as part of its settlement agreement, will also now take advantage of this option.

Google will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other features and controls to keep Android
users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to build and grow their apps. Google also
helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging from tablets to TVs to autos and more. And
it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.
</item 67>

<item 68>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 68>

<item 69>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
“Maybe you ask a friend.”

After hearing so much about what it was like to be an SEO, I decided it was time to better understand what’s been going on from the perspective of the search
engine. Google was slow to allow someone to talk with me, possibly because of the giant PR clusterfuck that has been the company’s past year (accused by the
federal government of being a monopoly; increasingly despised by the public; losing ground to Reddit, TikTok, and large language models), so I decided to start
by meeting up with a chipper, charismatic man named Duane Forrester.

Forrester was at Microsoft from 2007 until 2015, where he helped launch and manage Bing, the perpetual underdog to Google’s domination of online search. Before
and after his time at Microsoft, Forrester worked as an SEO, so he sees the industry from both sides, like an aerospace engineer who spent a few years at the
Department of Defense, left for the private sector, and now is much better at winning military contracts.
</item 69>

<item 70>
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
</item 70>

<item 71>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
No, we’ve always turned down special deals just for Epic. We’ve always fought on the principal that all developers should be, you know, given the same
opportunities. One of the documents in evidence was a 2019 email I sent — right before Fortnite launched the Marvel season with all this amazing new content and
the Star Wars event featuring J.J. Abrams. Right before that, I sent an email to all the Google senior executives saying that we wanted to bring Fortnite to the
Google Play Store in time for that event, and we wanted them to allow us and all other developers to process certain payments and keep them all.

That was our proposal to Google in 2019. If Google had said yes to that, that would have been awesome for all developers — the Android ecosystem would have
become much, much stronger, and Google would be in a much better position in the smartphone industry than they are today. We would’ve never had a dispute
because the problem would have been solved.

It’s always been in Google’s power to solve this problem.
</item 71>

<item 72>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
When I finally manage to jump through the flaming rings necessary to be allowed to speak on the phone with Sullivan, albeit with a communications chaperone also
on the line, I find him angry and defensive. He’s annoyed that anyone would think his era at Google has been less transparent than Cutts’ was: “We have reams of
help documents!” he told me. “We have more people assigned to work with SEOs than we did when Matt worked here!”

Sullivan is mad that the public and the media don’t really understand what he considers to be basic precepts about how search works, leading him to adopt a
rather scolding tone online. He’s frustrated that people want to know every last detail about Google’s algorithm because even “if we listed all one thousand of
the ranking signals” and how much each was worth, he said, that wouldn’t actually help SEOs do their jobs better, anyway.

And most of all, Sullivan is pissed that people think Google results have gone downhill. Because they haven’t, he insisted.
</item 72>

<item 73>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 73>

<item 74>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
SBF allegedly tried to buy off Trump: The TC team’s been trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced entrepreneur
accused of orchestrating the collapse of cryptocurrency exchange FTX. But fascinating details about SBF’s political dealings are emerging from a book by Michael
Lewis, “Going Infinite,” that debuted on the first day of the trial, like SBF’s attempt to buy off Trump to get him to not run again for president.

Gmail fights back against spammers: Google this week announced a series of significant changes to how it handles email from bulk senders in an effort to cut
down on spam and other unwanted emails. The company says that, starting next year, bulk senders will need to authenticate their emails, offer an easy way to
unsubscribe and stay under a reported spam threshold.

TikTok tests an ad-free tier: TikTok is testing an ad-free subscription tier for some users.
</item 74>

<item 75>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 75>

<item 76>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray reassured me that I was not crazy. Google results today do feel different from how they felt just five or six years ago for two major reasons. The first was
Google’s response to the disinformation panic around the 2016 election, which involved questioning the notion that the most reliable information could be chosen
by a form of popularity, meaning how many links a site received from other sites. As a result, the algorithm seemed to change its approach to links, especially
when it came to news and sites offering legal, financial, or health advice, and instead paid more attention to what Google came to call E-E-A-T: experience,
expertise, authoritativeness, and trustworthiness.

“E-E-A-T has had a pretty big impact on what types of results you see,” Ray told me. She’s done extensive (and fascinating) research around how certain sites
have fared under these new guidelines: Urban Dictionary, down! Mayo Clinic, up! Some people consider EEAT part of what’s making results better than ever.
</item 76>

<item 77>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 77>

<item 78>
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
</item 78>

<item 79>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 79>

<item 80>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
A newly unredacted version of the multi-state lawsuit against Meta alleges a troubling pattern of deception and minimization in how the company handles kids
under 13 on its platforms. Internal documents appear to show that the company’s approach to this ostensibly forbidden demographic is far more laissez-faire than
it has publicly claimed.

The lawsuit, filed last month, alleges a wide spread of damaging practices at the company relating to the health and well-being of younger people using it. From
body image to bullying, privacy invasion to engagement maximization, all the purported evils of social media are laid at Meta’s door — perhaps rightly, but it
also gives the appearance of a lack of focus.

In one respect at least, however, the documentation obtained by the attorneys general of 42 states is quite specific, “and it is damning,” as AG Rob Bonta of
California put it. That is in paragraphs 642 through 835, which mostly document violations of the Children’s Online Privacy Protection Act, or COPPA.
</item 80>

<item 81>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.
</item 81>

<item 82>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 82>

<item 83>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 83>

<item 84>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
The links that pop up when they go looking for answers online, they say, are “absolutely unusable”; “garbage”; and “a nightmare” because “a lot of the content
doesn’t feel authentic.” Some blame Google itself, asserting that an all-powerful, all-seeing, trillion-dollar corporation with a 90 percent market share for
online search is corrupting our access to the truth. But others blame the people I wanted to see in Florida, the ones who engage in the mysterious art of search
engine optimization, or SEO.

Doing SEO is less straightforward than buying the advertising space labeled “Sponsored” above organic search results; it’s more like the Wizard of Oz projecting
his voice to magnify his authority. The goal is to tell the algorithm whatever it needs to hear for a site to appear as high up as possible in search results,
leveraging Google’s supposed objectivity to lure people in and then, usually, show them some kind of advertising. Voilà: a business model!
</item 84>

<item 85>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
In 2009, Sullivan was described as “the closest approximation to an umpire in the search world,” so when he published “A deep look at Google’s biggest-ever
search quality crisis” in 2017 and then took a job as Google’s public liaison for Search only a few months later, it felt to some SEOs as though a
congressperson working on gun safety legislation had quit to become an NRA lobbyist.

“There is a thread across the industry of people who believe that Google just made Danny an offer he couldn’t say no to, and it was designed essentially to take
his voice out of the conversation,” Forrester told me. “I don’t believe that’s the case,” he went on, but compared to Cutts, “I think that Danny specifically
stays out of a lot of public conversations because he is in those private conversations with businesses.”

Was all that really Google’s fault? Or the SEOs? Or was this about something deeper and more human: the will to exploit something so much we destroy it.
</item 85>

<item 86>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I don’t know if that was on behalf of you and fellow developers. Can you tell us what that email was actually about?

Google didn’t leak the email; they leaked their ridiculously biased summary of it to Abner Li at 9to5Google. Don Harrison testified at trial that he didn’t
think Google leaked to the press. And then he was presented with a document including Google’s press team and Sameer Samat, head of all Android, basically
summarizing the articles that appeared as a result of Google’s leaking of our plans to the press.

So I was wondering if you could tell me what the email actually said.

Oh, it’s in evidence! This was the one I mentioned! Right before Christmas-ish 2019, we were planning to launch the new Marvel season with a bunch of Star Wars
content and its awesome lineup. I asked Google’s executives to let Fortnite come to the Google Play Store using our own payment method and to open up, you know,
using their own payment methods to all developers, not just Epic.
</item 86>

<item 87>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 87>

<item 88>
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
</item 88>

<item 89>
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
</item 89>

<item 90>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Google spent its 45 minutes attempting to explain that its dominance over the Android app market isn’t anything nefarious but simply the natural outcome of
Google fiercely competing with the iPhone and its iOS App Store, where Google would like the court to believe that competition truly lies.

If Google can convince the jury of that, it could be a winning argument in the case — because obviously, Google doesn’t have a monopoly on app stores or phones
in general. “You cannot separate the quality of a phone from the quality of the apps in its app store, and that means Google and Apple compete against each
other,” began Google lead attorney Glenn Pomerantz.

But Google wound up spending much of its opening statement attempting to explain away its seemingly bad behavior as normal business practices and didn’t always
succeed out of the gate.
</item 90>

<item 91>
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
</item 91>

<item 92>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 92>

<item 93>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 93>

<item 94>
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
</item 94>

<item 95>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 95>

<item 96>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 96>

<item 97>
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
</item 97>

<item 98>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 98>

<item 99>
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
</item 99>

<item 100>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 100>
=== END ITEM id=c79851dcd5810261 ===

=== ITEM id=41d3749e63caefb3 ===
Gold sentences and facts:
Sentence 0: Because GPT-4 is more likely to follow the instructions of “jailbreaking” prompts, the co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
  Fact 0.0: GPT-4 is more likely to follow the instructions of 'jailbreaking' prompts.
  Fact 0.1: The co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
Sentence 1: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 1.0: The entity's earlier actions were focused on concerns about the spread of illegal content.
  Fact 1.1: The entity's earlier actions were focused on concerns about the spread of disinformation related to the Israel-Hamas war.

Retrieved items:

=== END ITEM id=41d3749e63caefb3 ===

=== ITEM id=c32afcddd6e8be86 ===
Gold sentences and facts:
Sentence 0: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 0.0: The party referred to as 'it' did not release the full model, Gemini Ultra.
  Fact 0.1: Only a 'lite' version called Gemini Pro was released.
Sentence 1: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 1.0: The Justice Department is focused on the deals Google makes.
  Fact 1.1: Google makes deals with Apple.
  Fact 1.2: Google makes deals with Samsung.
  Fact 1.3: Google makes deals with Mozilla.
  Fact 1.4: Google makes deals with many others.
  Fact 1.5: The Justice Department is focused on ensuring Google is the default search engine on practically every platform.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:

=== END ITEM id=c32afcddd6e8be86 ===

=== ITEM id=46e2d272a64a8c5e ===
Gold sentences and facts:
Sentence 0: We identify with his struggles and his little glimmers of connection and triumph — which makes him the perfect superhero for Thanksgiving.
  Fact 0.0: People identify with the superhero's struggles.
  Fact 0.1: People identify with the superhero's little glimmers of connection and triumph.
  Fact 0.2: Identifying with the superhero makes the superhero the perfect superhero for Thanksgiving.
Sentence 1: Apart from the food - arguably the main component of the day - the holiday is a beloved time of year when Americans come together to celebrate what they are thankful for, either with family or friends.
  Fact 1.0: Thanksgiving is a holiday.
  Fact 1.1: The food is arguably the main component of Thanksgiving day.
  Fact 1.2: Thanksgiving is a beloved time of year.
  Fact 1.3: Americans come together during Thanksgiving to celebrate what they are thankful for.
  Fact 1.4: Americans celebrate what they are thankful for with family or with friends during Thanksgiving.

Retrieved items:
<item 1>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas is an excuse, new lands are wonderful



I definitely support Israel pic.twitter.com/AbLdeazQjR October 7, 2023

Israeli Prime Minister Benjamin Netanyahu said that since this morning, Israel had been at war and its first objective was to clear out the hostile forces that
infiltrated its territory. "The second objective, at the same time, is to exact an immense price from the enemy, within the Gaza Strip as well," he said, adding
that the third objective was to reinforce other fronts so that nobody should mistakenly join this war. "We are at war. In war, one needs to be level-headed. I
call on all citizens of Israel to unite in order to achieve our highest goal – victory in the war," Netanyahu said.

Since this morning, the State of Israel has been at war. Our first objective is to clear out the hostile forces that infiltrated our territory and restore the
security and quiet to the communities that have been attacked.
</item 1>

<item 2>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"The Israelis must be totally losing it right now."

the strategic planning that must have gone into this against one of the most high tech occupation and surveillance systems in the world will be studied by many,
the Israelis must be totally losing it right now https://t.co/oUX2rokEjh — Rania Khalek (@RaniaKhalek) October 7, 2023

Tarik Cyril Amar said: "And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed."

And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed. — Tarik Cyril Amar (@TarikCyrilAmar) October 7,
2023

Mjoachim, another user, said that Israel's defence forces have lost their fame. He said the IDF lost an army base, a general was captured right out of his
house.
</item 2>

<item 3>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"How come they never knew Hamas militants planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate heavily? We wait to see how it goes
but it will be messy."

Mossad is overated how comes they never knew this Hamas militants and had planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate
heavily?We wait to see how it goes but it will be messy. pic.twitter.com/W5vpoylus2 — Ben Muasya. (@muasya_ben) October 7, 2023

Another user said that no one can carry out such a major attack against Israel without intelligence support. "Who gave Hamas Intelligence? It was as if the
entire Israeli army and the Mossad were asleep."
</item 3>

<item 4>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
The second objective, at the same time, is to exact an… pic.twitter.com/MzKs7tfv4M — Benjamin Netanyahu - בנימין נתניהו (@netanyahu) October 7, 2023

Reacting to this, however, a user said that knowing the proficiency of the Mossad he found the idea of a "surprise attack" to be a willing lie.

Well, you got the war you wanted Beni...at what cost of Israeli lives you don't care. Knowing the proficiency of the Mossad I find the idea of a "surprise
attack"to be a willing lie on your side of this. https://t.co/35qRvgBJt2 — timm (@timmainsson) October 7, 2023

Ben Muasya, a journalist, said that Mossad is overrated.
</item 4>

<item 5>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
On Saturday, October 7, for example, suspicious movement in Gaza convinced the Shin Bet head to send a special team to the southern border, but he was unable to
convince the top IDF echelons to prepare accordingly. To handle those weak signals, Eiland claims, one must use "scenario theory", meaning, taking into account
scenarios that have a low probability of occurring, but with devastating results should they occur, and the simple way to prepare for them.

Eiland explains: "You detect certain noises on Friday night, but conclude that the probability of an attack is low, let's say only 10%. Those who act only on
the probability do not consider the question: if this attack happens - will you be able to live with yourself for eliminating this threat? There’s also a third
consideration, which is ease of effort: what is the effort required of me to prepare for this unlikely but dangerous scenario?
</item 5>

<item 6>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
</item 6>

<item 7>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
BHARAT is not immune to #terrorism and internal enemies @HMOIndia @PMOIndia @AmitShah — Dr. Jaswant Gandhi 🇮🇳 (@JaswantDr) October 7, 2023

However, some speculated that Mossad knew what was going to happen but let Hamas make a mistake so that Israel could launch a ferocious attack on it. "Mossad
was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, but they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful," a user said.

#Mossad was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful
</item 7>

<item 8>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
"Despite the clear threat posed by Gaza, the decision makers preferred to avoid a policy that would mean returning a large part of the country's population to
the reality of air raids and disrupted daily routines. The bottom line: the Iron Dome did not change the essential strategic situation; as firepower from the
Gaza Strip developed, so did the terrorist organizations’ ability to control daily life in Israel’s hinterland, hold it hostage, and restrain Israel from taking
proactive measures against them."

"Why didn't the relative success of the Iron Dome translate into the even more important strategic achievement of restoring Israel's freedom of action against
Hamas in the Strip?" asks Ortal rhetorically, and names several factors. "The first, like any other operational system, Iron Dome is also not perfect. The enemy
studies it and develops its own challenges against it, and the decision makers recognized the fact that the State of Israel is not hermetically protected.
Second, the enormous cost involved in deploying and operating [Iron Dome] batteries significantly impedes a policy based on continuous rounds of violence.
</item 8>

<item 9>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
He said the IDF lost an army base, a general was captured right out of his house. "Invincible "Merkava MK4 MBT's captured, and destroyed...what is wrong with
the Israeli security and defence forces? And (what) MOSSAD did?"

Dr Jaswant Gandhi, a professor of management, said that attack on Israel was a "great Mossad failure". He said India had to learn a lot and keep all
intelligence agencies duly greased and internecine conflict-proof. "Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence
Agencies duly greased and internecine conflict-proof. BHARAT is not immune to #terrorism and internal enemies," he wrote on X.

#Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence Agencies duly greased and internecine conflict proof.
</item 9>

<item 10>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas' surprise attack on Israel has shocked many who are now questioning the country's famed intelligence agency Mossad, which is known for its
intelligence-gathering capability. For some, the belief that Mossad was among the best in the spying world was shattered today when Hamas carried out the
biggest coordinated attack in Israel and captured some territory in southern Israel. It is not the attack that has surprised people but the scale at which Hamas
operated and got control of some military bases and soldiers.

Also Read: Israel at war: Hamas cells, military compounds destroyed, US backs Tel Aviv; all you need to know

Younis Tirawi, who reports on Palestinian Affairs, said that Hamas militants had taken over Sderot police HQ and armored vehicles in the town. An analyst wrote
that some of the images coming out of Sderot over the last few hours were reaching degrees of horror "that I don't really think I can use words to properly
describe.
</item 10>
=== END ITEM id=46e2d272a64a8c5e ===
