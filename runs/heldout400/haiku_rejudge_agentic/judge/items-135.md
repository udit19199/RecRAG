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

=== ITEM id=e3315b7f56f4e5e2 ===
Gold sentences and facts:
Sentence 0: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 0.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.
  Fact 0.1: The three categories are relevance, distance, and prominence.
Sentence 1: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 1.0: Google gets other benefits from Android.
  Fact 1.1: Google could solve this problem today if Google wanted to.
Sentence 2: This whole line of argument is called “market definition.” If the court decides the relevant market is phones and app stores, not specifically Android ones, then Google’s on the road to victory.
  Fact 2.0: The whole line of argument is called "market definition."
  Fact 2.1: If the court decides the relevant market is phones and app stores, not specifically Android ones, then Google is on the road to victory.
Sentence 3: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 3.0: The case was filed by the Arkansas-based publisher Helena World Chronicle.
  Fact 3.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 3.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 3.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: Vertical search, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Rosenberg, mobile
app distribution agreement, Oppo, Chrome, Play, revenue share agreements, best in class, Yahoo, Firefox, Android, Search, Apple
</item 1>

<item 2>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Expedia, Jeff Hurst, The Seattle
Times, search engine quality, European antitrust fine, European Union antitrust order, Yahoo, Firefox, Android, antitrust, TechCrunch, DOJ, European Union
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: Vertical search, Pre-installs, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, RSAs,
Android devices, Expedia, ads, Barry Diller, Sundar Pichai, Google travel listings, Barry Diller testimony, Expedia execs, organic listings, Yahoo
</item 3>

<item 4>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: Vertical search, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, LifeLock, Jamie
Rosenberg, Equifax data breach, iPhone 8, iPhone 8 official release, identity theft protection, competition between Google and Apple, Yahoo, Firefox, Google
Play, Android, Apple
</item 4>

<item 5>
Graph query row:
entity: Google Nest
relationship: MENTIONS
related: 200+ of the best deals from Amazon's Cyber Monday sale
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- 5 things we learned so far about the Google antitrust case -MENTIONS-> FTC-Facebook case
- Amit Mehta -PARTICIPATED_IN-> US v Google trial

Entities: Amazon antitrust lawsuit, FTC-Facebook case, Google antitrust lawsuit with Match Group, Google trial with Epic Games, Anticompetitive and unfair
strategies to maintain monopoly power, Vertical search, Pre-installs, Third-party apps, Commission structure, More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo, US v Google trial, Yahoo, Firefox, anticompetitive behavior, Google Play
</item 6>

<item 7>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Wired opinion article, Megan Gray, Is Google Search better than the rest? And is that fair?, Walker, The Age, US, Google, trial, WIRED
</item 7>

<item 8>
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

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- Amit Mehta -SAID-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit

Entities: Amit Mehta, Vertical search, Pre-installs, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google
trial, iPhone 8 plus, Auto insurance, Car insurance, Cheap flights, Car insurance quotes, DirecTV, Online colleges, AT&T, Spectrum, Comcast
</item 8>

<item 9>
The restaurant nearest Google (The Verge, 2023-10-26T13:00:00+00:00)
Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence. To improve local
ranking, Google encourages businesses to provide a trove of information on their profile. Owners should have basics like updated hours of operation and accurate
location details, but they should also respond to reviews, add photos, and even post individual items in stock to their Google page.

“I don’t think they’re really great marketing campaigns”

The prominence measure in Maps includes data from the web — links to the business, how the business ranks in traditional search, and press coverage, for
example. In this way, SEO impacts how a business appears in local search results, and news articles and links are one way Google tries to determine the
relevance of a page. (It’s why reporters’ inboxes are overrun with spammy requests to buy “guest posts” on our sites from marketers.)

Graph facts:
- The people who ruined the internet -MENTIONS-> SEO
- Duane Forrester -MENTIONS-> SEO
- Cade Lee -SAID-> SEO
- Missy Ward -SAID-> SEO
- Jairo Bastilla -SAID-> SEO

Entities: SEO, The restaurant nearest Google, Maps, relevance, distance, prominence, local search, traditional search, press coverage, Google, The Verge
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Chrome, Expedia, Yahoo, Firefox,
Android OEMs, antitrust, Microsoft, TechCrunch, Google, Justice Department, iPhones, U.S.
</item 10>
=== END ITEM id=e3315b7f56f4e5e2 ===

=== ITEM id=5190b9dcced6f2c2 ===
Gold sentences and facts:
Sentence 0: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 0.0: Judge Lewis Kaplan intervened to ask a question.
  Fact 0.1: Judge Lewis Kaplan asked if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance.
  Fact 0.2: Bankman-Fried admitted he had been told by Yedidia about that money.
  Fact 0.3: Bankman-Fried's admission came only after Judge Lewis Kaplan intervened.
Sentence 1: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 1.0: Wang testified that he heard Bankman-Fried tell another trader something.
  Fact 1.1: Bankman-Fried told another trader that it was okay to continue withdrawing money from the account.
  Fact 1.2: The condition was that the amount withdrawn was less than FTX's total trading revenue.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 2.2: The defense argued that the FTX founder acted in good faith.
  Fact 2.3: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.4: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison

That “basically confirmed my suspicions that had been rising all day” that Bankman-Fried had purloined the funds, Sun said. He ran Bankman-Fried through
possible explanations, including a margin loan program, and explained that none of them could fit the actual situation — for instance, Alameda had borrowed more
from FTX than had ever been in the margin loan program. He said Bankman-Fried acknowledged their conversation with a “yup, yup.” “I was expecting a bigger
response, but it was very muted,” Sun said.

After that, Singh spilled the beans: Alameda had withdrawn assets that included the customer funds, Sun testified. He quit the next day. Like Yedidia, Sun
testified with immunity.

In the courtroom, we were treated to a Good Morning America appearance by Bankman-Fried on December 1st, 2022.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda Research

Entities: Bankman-Fried, Alameda, In the end, the FTX trial was about the friends screwed along the way, Ellison, Singh, Sun, margin loan program, Good Morning
America appearance, customer funds, purloining funds, FTX trial, FTX, Wang, Yedidia, The Verge
</item 1>

<item 2>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
When FTX went down, Wang left himself with no plausible deniability. I cannot for the life of me understand why.

Normal customers — the defense objected to “normal” but was overruled — could be automatically liquidated

Today, the prosecution grilled Wang on a single, damning column in FTX’s databases. Called “allow_negative,” it allowed Alameda to have a negative balance.
Alameda could withdraw money even when its accounts didn’t have any, and it had an enormous line of credit. Wang added that it could place orders faster than
other users, and we already saw evidence that users may have unwittingly deposited money into Alameda rather than FTX.

The alleged motivation for this — as Wang had to explain at painful length to a jury — was avoiding liquidation. Liquidation is a way for FTX to manage risk
when people were engaging in risky bets on its platform. The cartoon version of futures trading is this: a futures exchange is the middleman that lets strangers
bet with each other.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, risk management, allow_negative, FTX, Wang, prosecution, defense, liquidation, futures trading, FTX platform, code review, jury, Today the
FTX jury suffered through a code review, The Verge
</item 2>

<item 3>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
But once it was implemented, the ability to go negative was used for other things including trading and unlimited withdrawals of FTX’s trading fees and customer
money.

I am summarizing briefly because Wang had to explain to the jury the difference between a front end and back end, explain futures trading, and read a bunch of
stuff into the record, which took a while. It was a tough day for the jury — and probably for Wang, who spoke quickly, perhaps from nerves.

The prosecution doesn’t just allege that Alameda had special, secret privileges — it claims they were used to obfuscate basic elements of FTX’s operations that
exposed its supposed selling points as a lie.

For instance, FTX advertised that it had a good liquidation system. In many crypto networks, if someone lost enough money, the exchanges could stick other
traders with the losses too. FTX touted its automated liquidation as a way to avoid this, so that one customer going bankrupt wouldn’t affect the others.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> the jury
- the jury -PARTICIPATED_IN-> Gary Wang's testimony
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda

Entities: the jury, Alameda, FTX, Wang, futures trading, the prosecution, crypto networks, automated liquidation system, Today the FTX jury suffered through a
code review, The Verge
</item 3>

<item 4>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
He did. But — I did hear these words uttered aloud in a court of law this morning, I am not creative enough to make this kind of thing up — his employees told
him “they were busy and I should stop asking questions because it was distracting.”

Yedidia — Bankman-Fried’s college friend, Bahamas roommate, and employee at FTX — had testified that he’d asked Bankman-Fried about the $8 billion hole on a
padel tennis court in their luxury complex in June or July. Today, in testimony, Bankman-Fried seemed to be trying to deny that conversation had ever taken
place. It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that
Bankman-Fried admitted he’d been told.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Lewis Kaplan
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Lewis Kaplan
- Lewis Kaplan -SAID-> market manipulation
- Lewis Kaplan -PARTICIPATED_IN-> evidentiary hearing
- Lewis Kaplan -PARTICIPATED_IN-> fraud trial of Sam Bankman-Fried

Entities: Lewis Kaplan, Sam Bankman-Fried didn’t ask where the $8 billion went, the $8 billion hole, Bahamas, FTX, Yedidia, Sam Bankman-Fried, The Verge
</item 4>

<item 5>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
In it, Bankman-Fried tried to justify the missing funds through the margin loan program. It was such a flimsy excuse that even George Stephanopoulos could see
through it.

Did Bankman-Fried really think he could get away with it all? He must still think so, or he would not be dragging his closest companions’ names through the mud
in a jury trial. Throughout the entire prosecution’s case, the defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison.
The hits the defense made on Singh didn’t make me doubt the overall truth of his story — they just made me think that Singh was kind of self-serving and trying
to secure himself a lenient sentence.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda Research

Entities: Bankman-Fried, In the end, the FTX trial was about the friends screwed along the way, Ellison, Singh, missing funds, FTX trial, margin loan program,
George Stephanopoulos, the defense, FTX, Wang, Yedidia, the prosecution, The Verge
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.

Graph facts:
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> cryptocurrency
- The people who ruined the internet -MENTIONS-> cryptocurrency
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> cryptocurrency
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> cryptocurrency
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> cryptocurrency

Entities: cryptocurrency, FTX trial, Southern District of New York, Christopher LaVigne, Withers, transactions, Bankman-Fried’s tweets, indictment, The FTX
trial is bigger than Sam Bankman-Fried, FTX, Sam Bankman-Fried, The Verge
</item 6>

<item 7>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda Research
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda Research
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda Research
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda Research
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda Research

Entities: Alameda Research, bankruptcy proceedings, wire fraud, conspiracy charges, scams, In the end, the FTX trial was about the friends screwed along the
way, FTX trial, FTX, Sam Bankman-Fried, The Verge
</item 7>

<item 8>
Graph query row:
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
published_on: 2023-10-06T17:22:02+00:00
publisher: TechCrunch
passage: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
matched_entity: Paying Chinese officials $150 million to restore access to $1 billion in frozen trading accounts
relationship: MENTIONS
connected_entity: Josh Naftalis
connection: MENTIONS
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Caroline Ellison, Bankman-Fried, Alameda Research, wire fraud, Nishad Singh, embezzlement, money laundering, conspiracy, political
donations, In the end, the FTX trial was about the friends screwed along the way, Joe Bankman, FTX trial, FTX, Yedidia
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Gary Wang, Caroline Ellison, Alameda Research, defense, Nishad Singh, Signal, FTX trial, The FTX trial is bigger than Sam
Bankman-Fried, LaVigne, witness tampering, FTX, Sam Bankman-Fried, government, Twitter
</item 10>
=== END ITEM id=5190b9dcced6f2c2 ===

=== ITEM id=3593c15f4d5d1574 ===
Gold sentences and facts:
Sentence 0: During Prime Day this past summer, Amazon introduced invite-only deals—a system to help make it easier for Prime members to access deals that are expected to sell out quickly—and the company brought it back for Prime Big Deal Days.
  Fact 0.0: Amazon introduced invite-only deals during Prime Day this past summer.
  Fact 0.1: The invite-only deals are a system to help Prime members access deals that are expected to sell out quickly.
  Fact 0.2: The invite-only deals system makes it easier for Prime members to access deals expected to sell out quickly.
  Fact 0.3: Amazon brought the invite-only deals back for Prime Big Deal Days.
Sentence 1: You can get this two-pack at Amazon for $15.99, so you can keep one at home and then put the other in a bag you bring around during everyday travels.
  Fact 1.0: A two-pack is available at Amazon for $15.99.
  Fact 1.1: The two-pack can be kept one at home and the other in a bag brought around during everyday travels.
Sentence 2: “Selling on Amazon is a life-changing opportunity and the amount of sellers that throw stones at the platform is astounding,” one user wrote.
  Fact 2.0: One user wrote that selling on Amazon is a life-changing opportunity.
  Fact 2.1: One user wrote that the amount of sellers that throw stones at the platform is astounding.

Retrieved items:
<item 1>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
The lengthy USB-C cable is capable of delivering 100 watts of power to beefy laptops like the 16-inch MacBook Pro, but you can also use it to power up
smartphones, tablets, and other gadgets.

for $7.99 ($12 off) at Amazon. The lengthy USB-C cable is capable of delivering 100 watts of power to beefy laptops like the 16-inch MacBook Pro, but you can
also use it to power up smartphones, tablets, and other gadgets. PopSockets is taking 25 percent off various grips, mounts, cases, and other accessories,
including this PopGrip for MagSafe phone handle / stand (on sale for $22.50) which is compatible with iPhone 12 phones and newer.

phone handle / stand (on sale for $22.50) which is compatible with iPhone 12 phones and newer. The worst thing about AirTags is you need something to stick them
in. Here’s a pack of four AirTag keychain holders for $8.91.

Graph facts:
- The best Black Friday deals you can get for under $25 -MENTIONS-> Bowser
- The best Black Friday deals you can get for under $25 -MENTIONS-> OLED TVs
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -MENTIONS-> PopSockets
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> game consoles

Entities: PopSockets, PopGrip for MagSafe phone handle / stand, AirTag keychain holders, The best Black Friday deals you can get for under $25, AirTags, 16-inch
MacBook Pro, USB-C cable, Amazon, iPhone 12, The Verge
</item 1>

<item 2>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Whenever you come back to your workstation with your MacBook, all you have to do is plug it in and you instantly have a multi-monitor setup. With an output of
up to 60 watts, you can use the dock to charge your laptop too.

Claiming the top spot in our Best USB Flash Drives guide, the SanDisk Extreme Pro balances speed, reliability, and price. The sleek aluminum case has a loop for
attaching it to a keyring and a slider to push out the USB-A plug. It is fast, performs reliably (we've been using one regularly for two years), and comes with
a lifetime warranty.

Best Prime Day Tablet Deals

Photograph: OnePlus

The OnePlus Pad (8/10, WIRED Recommends) is one of the few Android tablets we think is a worthy iPad alternative. It offers good performance, has great battery
life, and an excellent 144-Hz, 11.6-inch LCD display.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> iPad
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> iPad
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> iPad
- Sarah Michelle Gellar hits out at Roblox after her son’s account gets hacked -MENTIONS-> iPad
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> iPad

Entities: iPad, WIRED, MacBook, Prime Day, SanDisk Extreme Pro, OnePlus Pad, USB Flash Drives, Android tablets, 78 Absolute Best October Prime Day Deals (2023)
</item 2>

<item 3>
Graph query row:
article: Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
publisher: CNBC
entity: Amazon fulfillment center
</item 3>

<item 4>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
How to find the best Prime Big Deal Days deals?

WIRED is the only publication posting about Amazon Prime Day. Wait, sorry, looks a handful of other sites are, too. But you should only read WIRED, because we
actually vet every deal and compare it to the actual street price instead of an absurd MSRP that you will never observe in the wild. Also, we review the
products we recommend and don't just send you Prime Day deals on junk that will break. It's only a deal if you need it and it's good!

Are other retailers running sales?

Amazon Prime Day has inspired many imitators and you'll find sales from competitors like Best Buy, Target, and the like. Some of these sales are great, and
we'll flag when they are, but others are worth holding off on until Black Friday and Cyber Monday, when WIRED will again be the Internet's only website with
deals posts.

Retailer Sale Pages

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -PUBLISHED_BY-> WIRED
- These Are Our Favorite Walmart Cyber Monday Deals -PUBLISHED_BY-> WIRED
- Wired opinion article -PUBLISHED_BY-> WIRED
- 14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds -PUBLISHED_BY-> WIRED
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -PUBLISHED_BY-> WIRED

Entities: WIRED, Target, 78 Absolute Best October Prime Day Deals (2023), Amazon Prime Day, Prime Big Deal Days, Black Friday, Cyber Monday, Best Buy
</item 4>

<item 5>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Amazon Prime Day Part II is here, and that means a fresh batch of Prime Day deals. Technically Amazon calls this Prime Big Deal Days, but like most people, we
think of it as Prime Day Deux. As usual, most of these Prime Day deals require a Prime membership, but you can snag a 30-day free trial to make the most of the
event. We've been combing Amazon's website to bring you the best discounts on laptops, tablets, kitchen and home gear, headphones, and plenty more.

We test products year-round and handpicked these deals. Products that are sold out or no longer discounted as of publishing will be crossed out. We'll update
this guide regularly throughout Prime Day by adding fresh deals and removing dead deals.

WIRED Featured Deals

Table of Contents

If you buy something using links in our stories, we may earn a commission. This helps support our journalism. Learn more.

Best Prime Day Amazon Device Deals

The discount will apply automatically during checkout once you meet the $40 order threshold on select products.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -PUBLISHED_BY-> WIRED
- These Are Our Favorite Walmart Cyber Monday Deals -PUBLISHED_BY-> WIRED
- Wired opinion article -PUBLISHED_BY-> WIRED
- 14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds -PUBLISHED_BY-> WIRED
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -PUBLISHED_BY-> WIRED

Entities: WIRED, Prime Day, Prime membership, Amazon website, laptops, tablets, kitchen and home gear, headphones, 78 Absolute Best October Prime Day Deals
(2023), Amazon
</item 5>

<item 6>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
What Are Invite-Only Deals?

During Prime Day this past summer, Amazon introduced invite-only deals—a system to help make it easier for Prime members to access deals that are expected to
sell out quickly—and the company brought it back for Prime Big Deal Days. The feature is only available on select products but it has spread across a variety of
categories and price points.

If a specific product is part of the Invite-Only Deals system, you'll see a “Request Invite” button on the right-hand side. All you have to do is click it for a
chance to buy the product at that sale price. However, it's important to note that there's no guarantee you'll receive the invite. You can learn more about the
program and how it works in our story on How to Shop Like a Pro During Amazon Prime Day.

When is Prime Day (Prime Big Deal Days)?

The second and presumably final Amazon Prime Day 2023 is happening right about.... now. Prime Day deals started flowing at 3 am EST on October 10 and will
continue through October 11.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -PUBLISHED_BY-> WIRED
- These Are Our Favorite Walmart Cyber Monday Deals -PUBLISHED_BY-> WIRED
- Wired opinion article -PUBLISHED_BY-> WIRED
- 14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds -PUBLISHED_BY-> WIRED
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -PUBLISHED_BY-> WIRED

Entities: WIRED, Prime Day, 78 Absolute Best October Prime Day Deals (2023), Invite-Only Deals, Amazon Prime Day 2023, How to Shop Like a Pro During Amazon
Prime Day, Amazon
</item 6>

<item 7>
14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds (Wired, 2023-10-19T15:56:15+00:00)
It's light, comfortable for a wide variety of hand sizes and shapes, and features a couple of extra buttons without becoming cluttered. It is designed with
esports professionals in mind, meaning it has nice, clicky buttons, hyper-fast response time, and solid wireless connectivity even around 10 feet away from the
receiver.

This gaming mouse has a comfortable chassis; extra thumb buttons; and, perhaps most importantly, a modest price. Even when it's not on sale, it's one of our
more affordable picks, and this deal just makes it even more accessible.

Other Tech Deals

Anker Prime Photograph: Anker

Anker is kicking off its Black Friday sale early with discounts and a buy-one-get-another-50-percent-off promotion. Make sure you use the listed code to snag
the discounted price for each product. We generally like most Anker gear, but standouts include the Anker Prime power bank and the Anker Nano Power Bank.
Anker's tiny 511 30-watt charger is also excellent.

Graph facts:
- The Israel-Hamas War Is Drowning X in Disinformation -PUBLISHED_BY-> WIRED
- These Are Our Favorite Walmart Cyber Monday Deals -PUBLISHED_BY-> WIRED
- 78 Absolute Best October Prime Day Deals (2023) -PUBLISHED_BY-> WIRED
- Wired opinion article -PUBLISHED_BY-> WIRED
- 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods -PUBLISHED_BY-> WIRED

Entities: WIRED, 14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds, Anker, gaming mouse, Black Friday sale, Anker Prime power bank, Anker Nano Power
Bank, Anker 511 30-watt charger
</item 7>

<item 8>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Older buildings especially suffer from what I can only describe as loose socket syndrome, where those very convenient modern boxy socket extenders with five
different USB and USB-C charging ports simply cannot stay in place and fall right out of the wall because they’re too heavy. After one (totally pleasant)
vacation where I had to charge my phone, watch, AirPods, and laptop in a weird corner of my room with the plug-in charging hub propped up on a strategically
balanced mountain of books and sham pillows, I bought this Anker combination extension cord / power strip. It’s not a full power strip — just a cube with a few
sockets along with USB and USB-C charging ports at the end of a five-foot cable. I’ve brought it on a few trips since then. It takes up extra space in my
suitcase but each time has left me feeling vindicated about the purchase.

The three prongs at the end of the cable are static, rather than folding flat for easy packing.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, Anker, Black Friday, Some of our favorite devices are on sale for Black Friday., Anker combination extension cord / power strip, The Verge
</item 8>

<item 9>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Amazon-branded products range from home essentials like paper towels and batteries to snacks, office supplies, over-the-counter medicines, and more. This deal
is an easy way to stock up on frequently-used items for cheap.

Gift card deals are only worthwhile if you'd be spending the money anyway. With brands like Doordash, Instacart, Fandango, and more featured in this sale,
chances are you can find a worthy discount. Each card has a unique coupon code listed on the product page. Enter it during checkout to save.

Photograph: Amazon

Amazon devices are almost always going on sale, but this is an especially nice deal since it comes with a free smart plug that typically sells for about $20. It
isn't the same exact model, but a similar Kasa plug is the top pick in our Best Smart Plugs guide. The Echo Dot (5th Gen) is one of our favorite Alexa speakers.
You can use the included smart plug to do things like ask Alexa to turn off your box fan or turn on a lamp.

Graph facts:
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> Alexa
- The best Black Friday deals you can get for under $25 -MENTIONS-> Alexa
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> Alexa
- Amazon brings conversational AI to kids with launch of ‘Explore with Alexa’ -MENTIONS-> Alexa
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Alexa

Entities: Alexa, WIRED, Doordash, Instacart, Fandango, Echo Dot (5th Gen), Kasa plug, smart plug, Best Smart Plugs guide, 78 Absolute Best October Prime Day
Deals (2023), Amazon
</item 9>

<item 10>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
You can just take it off to wirelessly charge your device.

This cable can do it all, which is why it's currently sitting at the top of our Best USB-C Cables guide. It can deliver 100 watts, transfer data at up to 40
Gbps, and serve an 8K display (or two 4K displays). It supports DisplayPort Alt Mode (DP Alt Mode), USB Power Delivery (USB-PD), and PPS, and it is
backward-compatible with all earlier standards. It's pricey, even with the discount, but it rarely dips this low. This is also the longest size; the shorter one
is also on sale.

The Creator Kit acts as the link between your phone and various mounting solutions. It comes with a magnetic locking mount that works with the Peak Design
SlimLink phone case and a few different mounting options including a Peak Design Arca quick-release plate, a GoPro mount adapter, and a 1/4-inch tripod mount.

Graph facts:
- The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more -MENTIONS-> AirPods
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> AirPods
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> AirPods
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> AirPods
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> AirPods

Entities: AirPods, iPad, Apple Watch, WIRED, phone, GoPro, 54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods, Peak Design, cable, Best USB-C
Cables guide, Creator Kit, magnetic locking mount, Peak Design SlimLink phone case, Peak Design Arca quick-release plate, GoPro mount adapter
</item 10>
=== END ITEM id=3593c15f4d5d1574 ===
