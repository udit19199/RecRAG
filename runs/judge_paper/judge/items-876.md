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

This file has 2 items.

=== ITEM id=4aaf836d3a24cfb8 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The Justice Department focuses on these deals to ensure Google is the default search engine on practically every platform.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: In the last roundup, the writers learned that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 1.1: In the last roundup, the writers learned that Google tried to have Chrome preinstalled on iPhones.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 2>

<item 3>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 4>

<item 5>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He speculated that the company didn’t want the public to think too hard about how search works because that awareness “encourages regulators and makes people
realize, ‘Oh, there’s a lot of money here.’”

There has always been advertising and polemics from cranks, scammers, and liars. But now we see this stuff surfacing alongside truth, and we can’t tell the
difference. We move through our lives with a greater sense of distrust and fear and insecurity. At the end of the day, it’s Google’s world, and the SEOs are
only living in it.

And as much as I might hate the way the SEOs who don’t follow Google’s rules have altered my online experience, the reality is that most people running a
company will break whatever rules they are able to get away with breaking.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
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
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 11>

<item 12>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 12>

<item 13>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 13>

<item 14>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 14>

<item 15>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 15>

<item 16>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Google’s software is constantly crawling the web and updating its central index, which itself is more than 100 million gigabytes in size. When you enter a
query, Search uses a few quick processes to decide the meaning of your question, including what kind of information you might be after. Then, beyond just
looking for sites with the words you typed in, the system uses machine learning to rank results according to what it thinks you’re looking for, prioritising
high quality and easy to use sites. If you allow it, Google does bring in some information it knows about you to further refine results, including your location
and web history. If you’re looking to turn this off, it’s in your Google account settings called “Web & App Activity”. Google says it does not infer personal
information like race or religion to shape results. Loading Google works with independent “search quality raters” to gauge the effectiveness of its processes,
and says it ran almost 900,000 quality tests in 2022.
</item 16>

<item 17>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
the list of blue links in search results that are not sponsored. “Ad keyword matching is a long-standing and well-known process that is designed to connect
people to relevant ads. A separate process, which has nothing to do with ads, is used to match organic results to a query,” he said. “It’s no secret that Google
Search looks beyond the specific words in a query to better understand their meaning ... If you make a spelling mistake, or search for a term that’s not on a
page but where the page has a close synonym, or if you aren’t even sure exactly how to search for something, our meaning matching systems help.” It is true that
some search queries result in more ads than others, and what ads you get can vary wildly depending on your request, which is the result of the keyword matching
Google uses to sell ads. But Google denies pushing users towards the most monetisable subjects.
</item 17>

<item 18>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Google is now making it the default sign-in method for all users.

2 sec, let me text you some cash: When questioned about Mastercard’s prospects in emerging markets such as India, Mastercard’s CFO Sachin Mehra praised UPI for
helping with digitization but voiced reservations about its commercial sustainability, saying it is an “incredibly painful experience” for ecosystem
participants.

Bravely browsing — or searching — for a new job . . . : Brave Software, the maker of Brave Browser and Search, confirmed that it has laid off 9% of its
workforce across departments.
</item 18>

<item 19>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 19>

<item 20>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 20>

<item 21>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
In the beginning, the internet was seen as an improvement on the spirit of the public library. Here was an opportunity to transcend the gatekeepers controlling
who could publish a book, allowing mankind to fully connect and share knowledge. Instead, we have ended up in a situation arguably worse than before, where
nearly all online information runs through a single company, which assumes a veneer of civic utility, of impassive authority, when it is very much not a neutral
entity.

“There were so many true believers at Google in the early days,” Cutts told me. “As companies get big, it gets harder to get things done. Inevitably, people
start to think about profit or quarterly numbers.” He claimed that, at least while he was there, search quality always came before financial goals, but he
believes that the public underestimates how Google is shaping what they see, saying, “I deeply, deeply, deeply believe search engines are newspaper-like
entities, making editorial decisions.”
</item 21>

<item 22>
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
</item 22>

<item 23>
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
</item 23>

<item 24>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The process of this test did remind me how much data Google collects — not just my searches but anything I say to a smart speaker, any search result or ad that
I click — which will be a point in favour of DuckDuckGo for some, even if it didn’t perform as well and is harder to read. The data collection does also mean
that Google (not just search but also mobile widgets and other services) personalises to me more effectively. Loading Bing has overhauled its layout over the
past year to be very graphic and augmented by generative AI explanations from its chatbot. I like the idea of being able to ask follow-up questions in searches
this way, but in practice it’s not really there yet. In a search for “kids BMX bike green” (Google did pretty well and DuckDuckGo ignored the green part), Bing
Chat offered a comparison of a few different options but they were too expensive. I asked, “any under $200?” and it replied with more options that were all over
$300.
</item 24>

<item 25>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 25>

<item 26>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 26>

<item 27>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 27>

<item 28>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 28>

<item 29>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
“Maybe you ask a friend.”

After hearing so much about what it was like to be an SEO, I decided it was time to better understand what’s been going on from the perspective of the search
engine. Google was slow to allow someone to talk with me, possibly because of the giant PR clusterfuck that has been the company’s past year (accused by the
federal government of being a monopoly; increasingly despised by the public; losing ground to Reddit, TikTok, and large language models), so I decided to start
by meeting up with a chipper, charismatic man named Duane Forrester.

Forrester was at Microsoft from 2007 until 2015, where he helped launch and manage Bing, the perpetual underdog to Google’s domination of online search. Before
and after his time at Microsoft, Forrester worked as an SEO, so he sees the industry from both sides, like an aerospace engineer who spent a few years at the
Department of Defense, left for the private sector, and now is much better at winning military contracts.
</item 29>

<item 30>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 30>

<item 31>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 31>

<item 32>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 32>

<item 33>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 33>

<item 34>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 34>

<item 35>
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
</item 35>

<item 36>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 36>

<item 37>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 37>

<item 38>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 38>

<item 39>
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
</item 39>

<item 40>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 40>

<item 41>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 41>

<item 42>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
You can’t just be the most powerful observer in the world for two decades and not deeply warp what you are looking at.

For the past 25 years, the internet as we know it has been almost entirely defined and controlled by Google. What the SEOs do matters for all of us on a daily
basis, distorting how we perceive the world in ways we can hardly begin to imagine or understand. Yet any money that any SEO has made is a fraction of a crumb
compared to Google’s 10-layer cake. The company brings in hundreds of billions of dollars a year, profits that skew Google’s choices and priorities. As Google’s
founders wrote back in 1997: “we expect that advertising funded search engines will be inherently biased towards the advertisers and away from the needs of the
consumers.”

At the end of the day, it’s Google’s world, and the SEOs are only living in it

There’s a reason why most countries around the world have libraries that are public institutions: information that is controlled by a private business will
always be subject to that business’s bottom line.
</item 42>

<item 43>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Enter Google. The company revolutionized search by evaluating websites based on links from other websites, seeing each link as a vote of relevance and
trustworthiness. The founders pledged to be a neutral navigation system with no ads: just a clean white screen with a search box that would bring people off of
the Google landing page and out to a helpful website as seamlessly as possible. Users quickly decided this link-based sorting methodology was superior to the
existing search engines, and by the end of 1999, Google was handling the majority of online queries.

“I was basically just spamming Facebook with cars and articles about cars and sending traffic to banner ads, and that turned into $120,000 a month.”

Mapstead, like many of the early practitioners of SEO, figured out how to adapt. Almost as soon as Google took over, a secondary market emerged for links. For a
few hundred bucks, a firm in India or the Philippines could provide thousands of links from blog networks built entirely for that purpose.
</item 43>

<item 44>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He realized that most search engines were just listing websites in order of how many times a search term appeared on the site and in its tags, so he focused on
stuffing his sites with keywords, resubmitting his URL to the search engines, and waiting for the results to change.

Mapstead started pulling in $25,000–$30,000 a month, working 12- to 14-hour days. “It was how long could I stay awake and how little life could I have because
this was more money than I could have ever imagined in my lifetime,” he told me. “It was like I won the lottery, and I didn’t know how long it would last.”

Around this time, in 1997, an Italian professor published a journal article about what he called Search Engines Persuasion. “Finding the right information on
the World Wide Web is becoming a fundamental problem,” he wrote. “A vast number of new companies was born just to make customer Web pages as visible as
possible,” which “has led to a bad performance degradation of search engines.”

Enter Google.
</item 44>

<item 45>
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
</item 45>

<item 46>
The restaurant nearest Google (The Verge, 2023-10-26T13:00:00+00:00)
Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence. To improve local
ranking, Google encourages businesses to provide a trove of information on their profile. Owners should have basics like updated hours of operation and accurate
location details, but they should also respond to reviews, add photos, and even post individual items in stock to their Google page.

“I don’t think they’re really great marketing campaigns”

The prominence measure in Maps includes data from the web — links to the business, how the business ranks in traditional search, and press coverage, for
example. In this way, SEO impacts how a business appears in local search results, and news articles and links are one way Google tries to determine the
relevance of a page. (It’s why reporters’ inboxes are overrun with spammy requests to buy “guest posts” on our sites from marketers.)
</item 46>

<item 47>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 47>

<item 48>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
In 2009, Sullivan was described as “the closest approximation to an umpire in the search world,” so when he published “A deep look at Google’s biggest-ever
search quality crisis” in 2017 and then took a job as Google’s public liaison for Search only a few months later, it felt to some SEOs as though a
congressperson working on gun safety legislation had quit to become an NRA lobbyist.

“There is a thread across the industry of people who believe that Google just made Danny an offer he couldn’t say no to, and it was designed essentially to take
his voice out of the conversation,” Forrester told me. “I don’t believe that’s the case,” he went on, but compared to Cutts, “I think that Danny specifically
stays out of a lot of public conversations because he is in those private conversations with businesses.”

Was all that really Google’s fault? Or the SEOs? Or was this about something deeper and more human: the will to exploit something so much we destroy it.
</item 48>

<item 49>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
It was easy: buy links that led to your site and watch your ranking in Google’s results rise.

I came to understand that, since the dawn of the internet, there have been people attempting to manipulate search and then people decrying those manipulations
as the end of search’s ability to be useful. It works in cycles. People doing SEO find loopholes in the algorithm; critics complain about search results; search
engines innovate and close the loopholes. Rinse, repeat.

Before our current moment of widespread disillusionment with online information, the rise of SEO had reached a breaking point multiple times. In 2003, as Google
approached the deadline to disclose pertinent business information leading up to its IPO, the company quietly released an update cracking down. By 2011, SEO was
once again oppressively pervasive. TechCrunch published a story called “Why We Desperately Need a New (and Better) Google,” which argued that “Google has become
a jungle: a tropical paradise for spammers and marketers.”
</item 49>

<item 50>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 50>
=== END ITEM id=4aaf836d3a24cfb8 ===

=== ITEM id=e6cedd25d069b543 ===
Gold sentences and facts:
Sentence 0: Back then, just like today, home prices had boomed for years before Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
  Fact 0.0: Back then, as today, home prices had boomed for years.
  Fact 0.1: Fed officials were ultimately forced to hike interest rates.
  Fact 0.2: Fed officials hiked interest rates aggressively in an attempt to fight inflation.
  Fact 0.3: Home prices had boomed for years before Fed officials hiked interest rates.
Sentence 1: Postponements of such reports could complicate things for the Fed, which has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.
  Fact 1.0: Postponements of such reports could complicate things for the Fed.
  Fact 1.1: The Fed has insisted it will make upcoming decisions on interest rates based on what incoming data say about the economy.

Retrieved items:
<item 1>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
They also continued to decelerate from earlier months. A second economic report from the morning likewise offered both encouragement and caution for financial
markets. It said slightly fewer US workers than expected applied for unemployment benefits last week. On one hand, it indicates a job market with few lay-offs,
which means strength for the overall economy. But it could also be adding fuel to keep upward pressure on inflation.

After the reports, the 10-year Treasury yield rose to 4.7 per cent from 4.56 per cent late Wednesday. The two-year Treasury yield, which more closely tracks
expectations for the Fed, climbed to 5.06 per cent from 4.99 per cent. Yields are still down slightly for the week so far. After jumping last week to their
highest levels in more than a decade, yields regressed following speeches that investors saw as hints that the Fed may not raise its main overnight interest
rate further.
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
</item 2>

<item 3>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 3>

<item 4>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Brent crude, the international standard, also weakened. The resumption of US student-loan repayments, meanwhile, may funnel more dollars away from the spending
by consumers that has helped to keep the economy afloat. The latest monthly update on the US jobs market is due next week, with a couple of important reports on
inflation coming the following week. Postponements of such reports could complicate things for the Fed, which has insisted it will make upcoming decisions on
interest rates based on what incoming data say about the economy. The Fed’s next meeting on rates ends on November 1. On Wall Street, Nike jumped 6.7 per cent
after reporting better profit for the latest quarter than analysts expected. Strength overseas helped it make up for some declines in North America.

Loading Blue Apron soared 134.5 per cent after the meal kit company said it was being bought by Wonder Group for $US13 per share in cash in a deal valued at
$US103 million ($160.2 million).
</item 4>

<item 5>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Infratil (down 3.6 per cent), Meridian Energy (down 2.7 per cent) and Resmed (down 2 per cent) were among the biggest large-cap decliners. The lowdown The
Australian sharemarket shook off a positive lead from Wall Street on Wednesday after hotter-than-expected inflation data raised the prospect of another interest
rate rise at the Reserve Bank’s next meeting, which would weigh on the equity market. Interest-rate sensitive sectors, including REITS and consumer companies,
were among the weakest on the local bourse as investors digested the news.

GSFM investment strategist Stephen Miller said the inflation data, taken with the October Reserve Bank board meeting minutes, which revealed the board had “low
tolerance for a slower return of inflation to target than currently expected”, made the possibility of an interest rate rise from the central bank in November a
“near certainty”. However, commodity prices supported mining companies as iron ore prices increased 2.9 per cent overnight, bolstering Australia’s iron ore
giants.
</item 5>

<item 6>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Energy was among the laggards after its more than 3 per cent rally on Monday. A measure of US-listed Chinese shares added 3 per cent amid reports that the Asian
nation is considering new economic stimulus. “Policymakers have begun to acknowledge a lesser need for further policy action given financial conditions have
tightened considerably after the recent surge in Treasury yields,” said Ben Jeffery at BMO Capital Markets. “This acknowledgment may have reduced angst around
the need for additional rate increases.” Loading Investors will be watching for any hints in the September Fed meeting minutes due on Wednesday that would
suggest the Fed may not follow through with the last rise indicated in its economic projections, according to Anna Wong at Bloomberg Economics. Two critical
coming indicators for the US economy — Thursday’s consumer price index and Friday’s University of Michigan consumer-sentiment survey — may give a more
definitive read, she noted.
</item 6>

<item 7>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
That growing understanding has sent yields in the bond market to their highest levels in more than a decade, which in turn has undercut prices for stocks and
other investments. Treasury yields rose again on Tuesday following a mixed batch of reports on the economy. The yield on the 10-year Treasury edged up to 4.55
per cent from 4.54 per cent late on Monday and is near its highest level since 2007. It’s up sharply from about 3.5 per cent in May and from 0.5 per cent about
three years ago. One economic report on Tuesday showed confidence among consumers was weaker than economists expected. That’s concerning because strong spending
by US households has been a bulwark keeping the economy out of a long-predicted recession.

Besides high interest rates, a long list of other worries is also tugging at Wall Street. The most immediate is the threat of another US government shutdown as
Capitol Hill threatens a stalemate that could shut off federal services across the country.
</item 7>

<item 8>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
The average 30-year fixed mortgage rate, the most common type of mortgage in the U.S., has soared from 3.8% in March 2022 to over 7.5% today as a result. This,
in turn, has slowed mortgage purchase applications and caused home sales to plummet, just as it did in the ’80s; but home prices, echoing the dynamic of that
era, have yet to collapse.

One of the key reasons for the resilience of the housing market in both of these periods is demographics. “Noticeably, demographics were favorable back then,
with baby boomers having entered the prime homebuying age,” Park and her team wrote Thursday, arguing millennials are in a similar position today. “Some sales
activity should be supported by millennials reaching the prime homebuying age, and single-family building permits have steadily held up. This can help the
housing market retain some of its momentum without falling apart.”
</item 8>

<item 9>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
It comes as Chinese President Xi Jinping stepped up support for the world’s second-biggest economy and major commodity trading partner for Australia, issuing
additional sovereign debt, raising the budget deficit ratio and even making an unprecedented visit to the central bank. Loading Elsewhere, on Wall Street, the
benchmark S&P 500 index closed 0.7 per cent higher ahead of tech giants Microsoft and Google’s parent company Alphabet reporting their results after the close.
The Dow Jones Industrial Average added 0.6 per cent and the tech-heavy Nasdaq 100 gained 0.9 per cent. Shares in Microsoft climbed 5 per cent in after-hours
trading after the tech company said its sales jumped 13 per cent to $US56.5 billion ($88.9 billion) and profits climbed 27 per cent to $US22.3 billion in the
three months through September.

Meanwhile, shares in Alphabet fell 6 per cent after hours, after the Google parent delivered a mixed result.
</item 9>

<item 10>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
It had been above 5 per cent, and at its highest level in more than a decade, during October. The yield on the two-year Treasury, which more closely tracks
expectations for the Fed, went on a jagged run following the economic reports. It fell from 4.61 per cent just before the reports’ release to 4.57 per cent and
then yo-yoed before easing back to 4.57 per cent.

Loading Traders widely expect the Federal Reserve to hold its key interest rate steady at its meeting next week, before potentially cutting rates in March,
according to data from CME Group. Fed officials have recently hinted that the federal funds rate may indeed already be at its peak. It’s above 5.25 per cent, up
from nearly zero early last year. But Fed Chair Jerome Powell and others have also warned Wall Street about being overzealous in its predictions about how early
a cut could happen. Lower yields have been one reason prices for cryptocurrencies have been rising recently.
</item 10>

<item 11>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Adjustable-rate mortgages can lead to higher default rates when interest rates rise, but they now represent less than 5% of total purchase and refinance loans,
compared with over 35% at the peak of the pre-GFC housing cycle.

“We reiterate that we do not expect another housing crash like 2008,” Park and her team concluded after presenting this evidence.

Is it a repeat of the ‘turbulence’ of the 1980s?

According to Bank of America, today’s housing market looks a lot more like the early 1980s than it does 2008. Back then, just like today, home prices had boomed
for years before Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
</item 11>

<item 12>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
The ECB faces an extra legal burden on bond holdings that comes with operating in a currency union of 20 countries. Concerns around illegally financing
governments and debt mutualization have already landed the central bank in court several times.

Mixing Policies

Low interest rates and large-scale QE programs allowed treasuries to borrow on the cheap to finance stimulus campaigns, protecting labor markets, businesses and
consumers from collapse. But the spending blowout throughout and since the pandemic — part critical emergency funding, part political need to show an
all-hands-on-deck approach in crisis — contributed to the latest outbreak in inflation.

While the same kind of pulling in the same direction is needed to restrain demand, many governments are concerned that if they tighten policy too hard, voters
will kick them out and replace them with populists or extremists. That’s reviving questions about whether central banks can deliver price stability all on their
own.
</item 12>

<item 13>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Stocks have struggled under the weight of soaring Treasury yields in the bond market. High yields undercut stock prices by pulling investment dollars away from
stocks and into bonds. They also crimp corporate profits by making borrowing more expensive.

The yield on the 10-year Treasury, which is the centrepiece of the bond market, pulled back from its highest level since 2007, down to 4.73 per cent from 4.80
per cent late on Tuesday. Shorter- and longer-term yields also eased to allow more oxygen for the stock market. Yields fell following a couple of reports
indicating a slowing US economy. The first suggested hiring by employers outside the government was much weaker last month than expected. On Wall Street, that’s
currently good news because a cooling job market could mean less upward pressure on inflation. That in turn could convince the Federal Reserve to take it easier
on interest rates.
</item 13>

<item 14>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The stock market has largely been taking its cue from the bond market recently, and weak results announced in the afternoon for an auction of 30-year Treasury
bonds sent yields of all kinds higher. Higher yields can knock down prices for stocks and slow the economy by making borrowing more expensive. Yields had
already been on the rise in the morning after a report that showed inflation at the consumer level was a touch higher last month than economists expected. That
raises worries about the Federal Reserve keeping its main interest rate high for a long time, as it tries to undercut inflation by knocking down investment
prices and slowing the economy. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it comes to what it will do on November 1.
The inflation report, though, also had some encouraging nuggets for financial markets underneath the surface. After ignoring prices for food and fuel, which Fed
officials see as a better predictor of where inflation is heading, prices that consumers had to pay last month were in line with expectations.
</item 14>

<item 15>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“Everybody has one eye on the Middle East conflict and one eye on what’s happening with bond yields. The decline in bond yields is the key driver today,” said
John Praveen, managing director & co-chief investment officer at Paleo Leon. Loading While the Fed’s dovish comments were helping stocks on Tuesday and
investors were being sanguine about the Middle East, Praveen said that view could change if, for example, the fighting spread to other countries in the region.
“If tensions escalate bond yields might decline further because they’re a safe haven, but equities would sell off in that instance because of increased
uncertainty and risk aversion,” he said.

The Dow Jones Industrial Average rose 0.4 per cent, to 33,739.30, the S&P 500 gained 0.5 per cent and the Nasdaq Composite added 0.6 per cent. All 11 major S&P
500 sectors were advancing with materials and consumer staples leading gains, while technology was the slowest gainer.
</item 15>

<item 16>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Housing market pessimists have been sounding the alarm for years about a pending crash in the U.S. residential real estate market. Even before the Federal
Reserve began hiking interest rates to fight inflation last year, pushing mortgage rates to a 23-year high this month, years of surging home prices left some
experts warning that the housing market was a massive bubble ready to pop.

However, despite a widely bearish outlook in the industry, most real estate veterans have avoided arguing that home prices will drop like they did during the
2008 crash that kicked off the Global Financial Crisis (GFC). And so far, that’s been a wise decision.

It took until June 2022 for national home prices to peak, even as mortgage rates soared and purchase applications plummeted amid the Fed’s first interest rate
hikes. And while prices fell more than 5% from that peak by January of this year, they returned to a record high in July.
</item 16>

<item 17>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
The Australian dollar was US64.20¢ at the close as the US dollar fell for a fifth straight day, its longest losing streak since July. ‘Everybody has one eye on
the Middle East conflict and one eye on what’s happening with bond yields.’ John Praveen, managing director at Paleo Leon The 10-year Treasury yield came off
its 16-year peak, on track for its steepest single-day drop since May, as trading resumed in the US bond market, which had been closed for a holiday on Monday.
Ten-year yields tumbled as much as 18 basis points to 4.62 per cent. Fed swaps currently show about a 60 per cent chance the Fed will stay on hold in December,
compared with 60 per cent odds on another interest rate rise by then, just a week ago. But while oil prices eased after Monday’s rally, investors were still
closely monitoring developments the Middle East.

Israeli air strikes attacked Gaza on Tuesday, razing entire districts in the densely populated and impoverished enclave.
</item 17>

<item 18>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Tweet of the day

Quote of the day “Our view is that in the high likelihood there is another cash rate increase on Tuesday, that is going to be the nail in the coffin for
consumer confidence,” said CreditorWatch chief economist Anneke Thompson, amid warnings from some economists that another rate rise, as some market watchers are
predicting, will dampen spending intentions just in time for Christmas. You may have missed The men who built pubs and liquor juggernaut Endeavour Group into
the world’s largest poker machine operator have triggered a public spat over the future of a business that they say has lost its way.
</item 18>

<item 19>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.
</item 19>

<item 20>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
This can help the housing market retain some of its momentum without falling apart.”

Bank of America economists believe that low inventory of existing homes for sale combined with solid home sales to millennials could help put a “floor beneath
home prices,” but that doesn’t mean there won’t be some near-term pain due to higher mortgage rates.

“With rates likely staying higher for longer, we are cautious of potential turbulence ahead,” the group warned. Bank of America has previously forecast 0% home
price growth and falling home sales for the full year 2023, but it didn’t offer a new prediction in its latest note.

Eventually, as inflation fades leading the Fed to cut interest rates, housing affordability will improve. “At that stage, we should see a more stable and
healthy housing market,” Park and her team wrote. “Until then, hang tight, it may be a bumpy ride.”

This article has been updated to correct a mistaken term used to refer to Jeseo Park.
</item 20>

<item 21>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
“The bond market is retracing,” he said. “There’s also been strength in resource stocks amid rumours that the Chinese government might launch a substantial
stimulus package.” Clark said interest-rate sensitive sectors climbed on increased probabilities of an easing in interest rates. “Consumer discretionary and
information technology companies are winners in lower-interest rate environments,” he said. As more annual general meetings kick off in the next few weeks,
Clark said he would be looking at trading updates, especially of companies that didn’t give solid guidance in their last update.

Elsewhere, Wall Street advanced overnight, with the S&P 500 and Nasdaq touching three-week highs as dovish comments from the Fed officials pushed Treasury
yields lower. Following remarks from other top Fed officials on Monday, Atlanta Fed President Raphael Bostic said the US central bank did not need to raise
interest rates any further, and that he saw no recession ahead. Oil edged lower, with West Texas Intermediate settling below $US86 a barrel following its
biggest rally since April.
</item 21>

<item 22>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Treasury Wine Estates (down 1.9 per cent) and Woolworths (down 1.2 per cent) dragged the consumer staples sector (down 1.1 per cent) lower. Pilbara Minerals
(down 2.1 per cent) and Qantas (down 1.5 per cent) were also among the biggest large-cap decliners. The lowdown Stronger iron ore prices helped to bolster the
local bourse on Monday but not enough to offset the prospect of higher-for-longer interest rates ahead of the Reserve Bank’s meeting on Tuesday. Iron ore
heavyweights advanced after the iron ore price increased 1.4 per cent overnight with lithium miner Liontown resisting the downward momentum of lithium prices,
which dropped to the lowest level in two years.

Most sectors traded lower but interest-rate sensitive sectors including healthcare and consumer companies were among the weakest amid expectations that the RBA
will keep rates on hold in October. The National Australia Bank has forecast another rate hike to 4.35 per cent in November.
</item 22>

<item 23>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
The National Australia Bank has forecast another rate hike to 4.35 per cent in November. Elsewhere, Wall Street closed out its worst month and quarter of the
year with more losses on Friday. The S&P 500 slipped 0.3 per cent and the majority of stocks within the index sank. The Dow Jones fell 0.5 per cent and the
Nasdaq composite edged 0.1 per cent higher. The Australian sharemarket is set to retreat, with futures pointing to a fall of 38 points, or 0.5 per cent, at the
open. Solid gains for stocks early on faded as pressure built from within the bond market. After easing earlier in the day on encouraging signals about
inflation, Treasury yields got back to rising as the day progressed. The yield on the 10-year Treasury yield returned to 4.58 per cent, where it was late
Thursday, after dipping to 4.52 per cent. It’s again near its highest level since 2007.
</item 23>

<item 24>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
“Westpac’s results were better than expected, especially because it’s been a bit of a perennial underperformer in the past few years,” he said, adding that it
set the tone for results from NAB and ANZ, which are due to report in the next week. Loading Clark also said there was a reversal of the interest-rate trade
seen in recent weeks. “Long-term bond yields came off quite significantly in the last few days following weaker than expected employment data on Friday, which
has seen some investors swivel back into growth stocks,” he said. The market has now switched from pricing in an 80 per cent chance of a rate rise on Tuesday to
50-50, Clark said.

On Wall Street, the S&P 500 climbed 40.56 points on Friday, or 0.9 per cent, to 4,358.3.
</item 24>

<item 25>
ASX rises after Wall Street gains on rate optimism; Oil drops (The Sydney Morning Herald, 2023-10-10T23:50:33+00:00)
Billionaire investor Paul Tudor Jones told CNBC the current geopolitical environment was the “most threatening and challenging” he’d ever seen following Hamas’s
attack on Israel over the weekend and predicted the US will enter into a recession early next year. The shekel regained its footing as the central bank duelled
short sellers to contain the market fallout from Israel’s conflict with militant group Hamas. European gas prices jumped, with Finland on high alert as it
suspects a gas pipeline leak in the Baltic Sea was caused by a deliberate act of destruction, fuelling concerns about the safety of Europe’s energy
infrastructure. Base metals sank amid fresh fears over property turmoil in China. Tweet of the day

Quote of the day “This is as strong a level of confidence in the market that we have seen since before COVID,” said The Leasing Agency group managing director
Terence Yeh as the return of international students and a growing number of apartments in inner-city Melbourne provides a shot in the arm for CBD hospitality
businesses.
</item 25>

<item 26>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Emerging market economies with large volumes of USD denominated debt are especially vulnerable as they could be caught in a vicious circle of their external
debt (in local currency terms) ballooning due to strong DXY and their servicing ability being impaired by tepid economic activity owing to high rates. Even
advanced economies have felt the heat, and the ECB etc. have maintained a hawkish narrative, Jain said.

Another developing story seems to be the greenshoots of recovery in China backed by domestic consumption in the Golden week, and tourism related to the Asian
Games. For now, these seem to buttressed on stimulus and base effects as China is now fully open.

The long-term trajectory remains to be seen. In any case, for now this has provided a floor for commodity prices, especially base metals. Fickle commodity
prices will also be victims of fresh geopolitical tensions in West Asia, which has pushed up Brent crude. Food prices could also play truant with climate change
affecting cropping area worldwide: sugar and rice prices are booming, the note said.
</item 26>

<item 27>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
It’s important to talk to the private sector, and to understand in every place and every industry what the main damage is and what they need in order to get
back to normal. It should be done professionally, preferably with no politicization of the recommendations."

What do you think the Bank of Israel should do in its next interest rate decision, on October 23?

"It’s still to early to tell how the war will affect the rate of inflation in the short and medium term. Just as there are new inflationary factors, such as the
depreciation of the shekel, there will also be factors working in the opposite direction, given the expected decline in private consumption, tourism, and other
items.

"As with fiscal policy, here too a country at war is allowed to deviate temporarily from the set inflation target. At the moment there is no surge in inflation
expectations, and the investment community is aware of the existence of index-linked assets and the exchange rate that can assist in hedging inflation risk.

"As for cutting the interest rate, it’s too early to consider that.
</item 27>

<item 28>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
That’s reviving questions about whether central banks can deliver price stability all on their own.

“If we were designing optimal policy arrangements from scratch, monetary and fiscal policy would both have a role in managing the economic cycle and inflation,
and that there would be close coordination,” Philip Lowe said in his last speech as Australian central bank governor in September.

Economists surveyed by Bloomberg predict fiscal policy will somewhat counteract the Fed’s efforts to rein in inflation in the US.

“It’s true that there are circumstances where working hand in hand and supporting each other has proved helpful,” ECB President Christine Lagarde told a panel
discussion in June at the institution’s annual economic forum.

Fed Chair Jerome Powell, who sat to her right, signaled he wasn’t ready to rely on that kind of cooperation. “Our assignment is to deliver price stability kind
of regardless of the stance of fiscal policy.”

Central bankers warn that any failure to scale back fiscal spending risks coming at the cost of yet higher interest rates.
</item 28>

<item 29>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
The rise in the consumer price index peaked at around 14% in 1980 before then–Fed Chair Paul Volcker’s hawkish policies sent mortgage rates to 18% in a year’s
time, cooling inflation, but sparking a recession. This caused a serious downturn in the housing market in which home sales and building levels cratered.
However, national home prices actually remained stable.

At the start of Volcker’s term as Fed chair in August 1979, the median U.S. home sales price was $64,700. And even after a near doubling of mortgage rates, that
figure rose to $69,400 by the second quarter of 1981.

Over the past 18 months, the Fed’s current chair, Jerome Powell, has been following a very similar game plan to Volcker’s, raising interest rates aggressively
to quash inflation.
</item 29>

<item 30>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
She recommends they be ready to react preemptively, even when inflation hasn’t yet spun out of control.

They may be called into action soon on that front, should an escalation in the conflict in the Middle East hit oil deliveries.

When the next big global slowdown comes, though, flexibility may be needed the other way. Europe’s eight-year experiment with negative rates ended with mixed
reviews last summer as to whether it was all worth it.

The Bank for International Settlements argues that there’s room for greater tolerance for moderate shortfalls even if they’re persistent, because “low-inflation
regimes, in contrast to high-inflation ones, have self-stabilizing properties.”

Rethinking Quantitative Easing

With a more flexible approach to those 2% targets, monetary policy after the 2008 financial crisis would have looked very different in many parts of the world.
</item 30>

<item 31>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
What Bloomberg Economics Says… “A long period of galloping price gains, and fears that the last yards back to target could be most painful for workers, have
reignited the debate about whether central banks should aim for a higher rate of inflation. That’s a conversation worth having. But for monetary policymakers,
the imperative of retaining credibility means the right time for it is after inflation is back at target, not before.” — Tom Orlik, chief economist

Rethinking Targets

So long as people believe prices will get back toward 2%, central bankers have some leeway in deciding how aggressive they need to be in pursuing that goal.

Economists covering 16 of the world’s most important central banks say policymakers will allow more time to bring inflation back to target if it means less
damage to their economies. The Bloomberg special survey also shows that a sizable minority sees them going even further, accepting price pressures that are
either slightly too strong or too weak — as long as expectations remain anchored.
</item 31>

<item 32>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The laggards On the losing end, energy companies (down one per cent) weighed down the index. Woodside (down 0.8 per cent), Santos (down 0.4 per cent) and Ampol
(down 1.2 per cent) dropped, along with coal miners Yancoal (down 1.7 per cent) and Whitehaven (down 2 per cent). Lynas Rare Earths (down 3.1 per cent) was the
biggest large-cap decliner, while industrials companies (down 0.2 per cent) including Computershare (down 2.9 per cent), Seven West Holdings (down 1.9 per cent)
and Worley (down 1.3 per cent) were also weaker. The lowdown

TMS Capital portfolio manager Ben Clark said the big news for markets on Monday was the trading update from Westpac, which he said led a resurgence in the
banking sector’s performance.
</item 32>

<item 33>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.
</item 33>

<item 34>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
After the most aggressive monetary-tightening campaign in four decades, academics and economic practitioners are running autopsies on what could have prevented
the cost-of-living crisis and how to ensure the same mistakes won’t be repeated.

Markets have scrambled to price in high-for-longer interest rates, with a new war in the Middle East adding yet more risk to an already uncertain outlook
confronting central bankers as they gather for their penultimate meetings of a tumultuous year.

The policy navel-gazing is centering around three debates. How much flexibility central banks can allow in reaching their inflation targets, the effectiveness
of asset purchases in the policy mix, and the merits of monetary and fiscal coordination.

Bloomberg surveyed economists from around the world to gather views on those three debates. Their verdict: Central banks won’t break their economies in a rush
to hit inflation targets, QE will be used more sparingly in the future, and fiscal policy risks countering the work of monetary authorities.
</item 34>

<item 35>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
housing market set off the dominoes that became the Great Financial Crisis that defined a whole decade economically—and in some ways, our current predicament.

In 2010, the last time existing home sales were so low, the global economy was on life support and the Fed slashed interest rates down to zero to revive the
flatlining patient—fueling an “everything bubble” that inflated all kinds of assets along the way, not the least, housing. The zero-rate regime persisted until
the highest inflation in 40 years surprised many economists and prompted the massive series of rate hikes that have brought us to the once-unthinkable 8%
mortgage.

Some onlookers saw what was coming, for instance Zillow, which warned in May that housing would enter a “deep freeze” if the debt limit standoff failed to
resolve and America defaulted, as that would send mortgage rates up to 8%. Of course, the default was averted, but here we have arrived at 8% mortgage rates
anyway.
</item 35>

<item 36>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Trillions of dollars, euros, yen and pounds of asset purchases did little to raise prices in the face of global disinflationary forces until governments used
the money they raised to stuff cash into consumers’ pockets during Covid lockdowns.

But that’s also been blamed for distorting financial markets. Episodes such as the Silicon Valley Bank blow-up are seen by some as a direct result of central
bank reserves creation under QE, along with regulatory and supervision failures.

Only 40% of economists surveyed predict central banks will use QE the same way as they did before. A quarter expect them to deploy it more sparingly, about 30%
see its only role going forward as a tool to address financial-stability concerns and a small minority doesn’t see it being used again at all.

There are other problems with bond-buying that may affect how it’s used in the future. QE effectively swaps long-term borrowing costs for short-term ones.
</item 36>

<item 37>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Olivier Blanchard, a former IMF chief economist, has long argued in favor of raising the inflation target, and former European Central Bank Vice President Vitor
Constancio has also embraced the idea. But it’s a controversial view and only possible from a position of credibility, which means central banks would likely
have to get inflation back to 2% first.

“It would be a mistake of the first order to think you can change a goal you have set if you can’t achieve it,” according to Bundesbank President Joachim Nagel.

Global trends suggest inflation will be stronger than in the past, with former Bank of England Governor Mark Carney among those saying rates won’t return to
pre-pandemic lows.

One lesson Gita Gopinath, the IMF’s No. 2 official, draws from the latest inflation episode is that policymakers mustn’t assume that looking through supply
shocks — as text books suggest — is the optimal response.
</item 37>

<item 38>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Dow Jones Industrial Average gained 222.24, or 0.7 per cent, to 34,061.32, and the Nasdaq composite jumped 184.09, or 1.4 per cent, to 13,478.28. Stocks
surged through the week on rising hopes that the Federal Reserve is finally done with its market-crunching hikes to interest rates, in order to get inflation
under control. A report on Friday underscored that pressure is easing on inflation after it showed employers hired fewer workers last month than economists
expected. It’s a stunning turnaround from just a week ago, when Wall Street was reeling after the S&P 500 had fallen 10 per cent below its high point for the
year. That sent Wall Street’s main index into what investors call a “correction.” Since then strong profit reports helped drive some stocks to towering gains.
Generac, a maker of back-up generators, soared nearly 28 per cent for its best week since its stock began trading in 2010.
</item 38>

<item 39>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
The lowdown

Hotter-than-expected inflation data and the prospect of more rate rises in the world’s biggest economy weighed on sentiment in the Australian sharemarket as
interest-rate sensitive sectors such as IT and consumer discretionary dragged the index lower on Friday. While commodity prices boosted iron ore players,
company specific news was behind some of the biggest swings. Resmed extended its losses after brokers at RBC downgraded the stock and gold miner Newcrest
stepped up after investors approved its merger with US gold mining giant Newmont. US stocks slipped on Thursday as the clamps tighten on Wall Street from rising
yields in the bond market. The S&P 500 was 0.7 per cent lower in afternoon trading and on track to break a four-day winning streak, its longest since August.
The Dow Jones Industrial Average was down 204 points, or 0.6 per cent, as of 3pm Eastern time, and the Nasdaq composite was 0.8 per cent lower.
</item 39>

<item 40>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
The laggards Real estate investment trusts (REITS, down 2 per cent) dropped further following the latest inflation data, with Mirvac (down 3.7 per cent) and
Dexus (down 3.3 per cent) sliding lower. Dexus chief executive Darren Steinberg on Wednesday announced he would step down from the position next year after 11
years at the helm. At the annual meeting in Sydney on Wednesday, the remuneration report received a first strike after failing to pass the 25 per cent voting
threshold. Loading Consumer staples (down 1.4 per cent) were also weaker with Woolworths losing 2 per cent despite sales across the company rising 5.3 per cent
in the 14 weeks to October, as it vowed to pass on easing costs for protein and fresh produce amid rising mortgage and rental pressures. In other company news,
shares in embattled funds manager Magellan dropped 3.5 per cent after the abrupt departure of its chief executive, David George.
</item 40>

<item 41>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
QE effectively swaps long-term borrowing costs for short-term ones. What’s been a lucrative deal for taxpayers when official interest rates were low has now
turned into a disastrous trade.

The clearest depiction of the problem is in the UK, where the BOE secured taxpayer indemnity for any losses on QE. Over the next decade, it estimates, its
purchases will cost the government over £200 billion ($243 billion).

And policymakers have little experience in unwinding their balance sheets, where small mistakes can trigger big market turbulence.

The Fed experienced some of that when it tried to shrink bond holdings between 2017 and 2019. More recent efforts to reduce portfolios have progressed rather
smoothly, partially because central banks have amassed so much debt over the years that they’re far away from any thresholds that would trigger a squeeze.

But the fact that they’re treating quantitative tightening as a technical adjustment rather than a part of their efforts to conquer inflation raises questions
about the future use of a tool that’s only trusted to work one way.
</item 41>

<item 42>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Bitcoin briefly topped $US35,000, while Treasury 10-year yields edged lower, following Monday’s intense volatility. Investors looking to the earnings season for
a dose of good news were hanging their hopes on big tech. The five biggest companies in the S&P 500 — Apple, Microsoft, Alphabet, Amazon and AI chipmaker Nvidia
— account for about a quarter of the benchmark’s market capitalisation. Their earnings are projected to jump 34 per cent from a year earlier on average,
according to analyst estimates compiled by Bloomberg Intelligence.

“As these big tech stocks go, so does the overall market,” David Trainer, chief executive officer of New Constructs, said before Microsoft and Alphabet’s
results. “Strong big tech earnings may just be what’s needed to end the stock market correction that started in late July. If big tech companies blow their
numbers out of the water and provide strong guidance for future earnings, then we could see the stock market rally strongly through the end of the year.”
</item 42>

<item 43>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Iron & steel have also witnessed accelerated long-term loan disbursements in the past six months. Infrastructure is witnessing an increase in credit flow with
67 per cent of the respondents indicating an increase in long-term loans as against 57 per cent in the previous round.

The survey suggests that the outlook on expectation on growth of non-food industry credit over next six months is optimistic with 42 per cent of the
participating banks expecting non-food industry credit growth to be above 12 per cent (as compared to 36 per cent in the previous round).

Rajan Jain, Head, Credit Research, SBI Caps, said in a note that whether the yield curve dynamics mean the US is moving to a structurally higher-rate economy
vs. post GFC normal remains a moot question, with important implications for investment demand and system risk.

These movements have had a cascading effect on global economies, which have seen routs in their respective bond markets, besides a fresh round of currency
depreciation.
</item 43>

<item 44>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
It’s again near its highest level since 2007.

Loading Treasury bonds are seen as some of the safest investments possible, and when they pay higher yields, investors are less likely to pay high prices for
stocks and other riskier investments. That’s a big reason why the S&P 500 dropped 4.9 per cent in September to drag what had been a big gain for the year down
to 11.7 per cent. Treasury yields have been climbing sharply as Wall Street accepts a new normal where the Federal Reserve is likely to keep interest rates high
for longer. The Fed is trying to push still-high inflation down to its target, and its main tool of high interest rates does that by trying to slow the economy
and hurting prices for investments. The Fed’s main interest rate is at its highest level since 2001, and the central bank indicated last week it may cut
interest rates next year by less than it earlier expected.
</item 44>

<item 45>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That will arrive on Friday. But “if Friday’s report also shows the labour market is cooling, stock investors may worry a little less about indefinitely higher
interest rates,” said Mike Loewengart, head of model portfolio construction at Morgan Stanley Global Investment Office. A second report on the economy said
growth in US services industries slowed in September by a touch more than economists expected.

It also offered some hints of sticky pressure on inflation, with prices paid by services companies rising last month at a similar rate as in August. Oil prices
tumbled on Wednesday to take some heat off inflation. Benchmark US crude fell $US5.01 to settle at $US84.22 per barrel for its worst drop in just over a year.
It’s been pulling back since, topping $US93 last week. Brent crude, the international standard, lost $US5.11 to $US85.81. Loading Prices for crude had been
generally charging higher from $US70 during the summer, following announcements of cuts to production by some oil-producing countries.
</item 45>

<item 46>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
Plus, the raising of interest rates in an effort to combat inflation is strikingly similar to the ’80s. Back then, Fed Chair Paul Volcker fought inflation
through aggressive interest rate hikes, with the average 30-year fixed mortgage peaking at about 18% by late 1981. Sound familiar? Current Fed Chair Jerome
Powell has set the tone for the most aggressive global hiking of rates in the modern era. And in due course, this week, mortgage rates hit 8%, the highest
they’ve been in more than 20 years.

All things considered, home sales activity also plummeted from 1978 to 1982. Existing-home sales dropped 50% during that period, according to the Office of
Policy Development and Research.
</item 46>

<item 47>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Rising rates have made already stretched big tech valuations look increasingly expensive, with the group remaining the most-crowded trade among fund managers,
according to Bank of America. That’s prompted some investors to pay up for protection against a sell-off in Alphabet and Microsoft — two of the handful of
heavyweights responsible for all of the S&P 500 Index’s advance this year. The pain in long-duration growth stocks, fuelled in recent weeks by a relentless
surge in Treasury yields, is finally on the verge of subsiding. The yield on 10-year Treasuries was little changed at 4.85 per cent overnight. Oil closed at the
lowest in more than a week amid signs the crude market’s tightness has slackened and the Israel-Hamas war will remain contained for the time being. West Texas
Intermediate extended Monday’s drop with a 2.2 per cent fall to $US83.59 a barrel, paring most of the gains made after Hamas’ attack on Israel on October 7.
</item 47>

<item 48>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.
</item 48>

<item 49>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
Such cuts can act like steroids for financial markets. For investors around the world, the “Fed matters more than other central banks,” and weak US data is “the
only game-changer for markets”, foreign-exchange strategists at Bank of America wrote in a BofA Global Research report. A separate report on Friday said growth
in US services industries, such as finance and construction, was weaker last month than economists expected. The report from the Institute for Supply Management
also suggested a slight easing in prices.

Excitement about a potentially easier Fed was more than enough to offset a fall for Apple, which is Wall Street’s most influential stock. The most valuable US
stock fell 0.5 per cent despite reporting stronger profit for the latest quarter than analysts expected. Analysts said investors were likely disappointed with
Apple’s forecast for revenue for the last three months of 2023.
</item 49>

<item 50>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Friday’s economic data shows that not only was inflation a touch cooler than expected in August, so was growth in spending by US consumers. That can be a
positive for inflation because it means not as many dollars are pouring into purchases. That in turn could give companies less encouragement to try to raise
prices further. But it may also dent what’s been a big driver keeping the US economy out of a recession.

“It came to a boil during a hot summer, and the temperature is really starting to come down,” said Brian Jacobsen, chief economist at Annex Wealth Management,
of spending growth by US consumers. “Higher energy prices, student loan debt repayments and real disposable incomes that have been on a declining trajectory
since June doesn’t bode well.” Oil prices have jumped to their highest level in more than a year, which is pressuring the economy by raising fuel costs for
everyone. A barrel of US crude sank 92¢ Friday to settle at $US90.79, but it’s still up sharply from $US70 in June.
</item 50>
=== END ITEM id=e6cedd25d069b543 ===
