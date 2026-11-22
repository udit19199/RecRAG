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

=== ITEM id=4b71df25c77bfdc0 ===
Gold sentences and facts:
Sentence 0: “Certainly there wasn’t a valid alternative to Google at the time,” Cue said.
  Fact 0.0: Cue said there certainly wasn't a valid alternative to Google at the time.
Sentence 1: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 1.0: Google could solve this problem today if it wanted to.
  Fact 1.1: Google gets other benefits from Android.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google “siphons off” news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that this is done through anticompetitive means.

Retrieved items:
<item 1>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 1>

<item 2>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
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
</item 10>
=== END ITEM id=4b71df25c77bfdc0 ===

=== ITEM id=c38c2aa7a91a4956 ===
Gold sentences and facts:
Sentence 0: But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
  Fact 0.0: The speaker thinks that people will look back and see that the past year is a really important turning point for the role social media plays in people's lives.
  Fact 0.1: The speaker adds this.
Sentence 1: The Goellers are now seeking support again, launching the "Save Lucas" campaign across social media.
  Fact 1.0: The Goellers are now seeking support again.
  Fact 1.1: The Goellers are launching the 'Save Lucas' campaign.
  Fact 1.2: The 'Save Lucas' campaign runs across social media.

Retrieved items:
<item 1>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The founders don’t regret their experience building Pebble, even though it didn’t turn out as they hoped, they said.

Says Cselle, “One thing that I learned is that there’s absolutely an audience that wants to see a new kind of Twitter-like platform built and will ask for the
features that Twitter has.” He suspects the team might stay together to work on something new after Pebble wraps, taking the learnings from Pebble with them.
That includes better communicating to users what great content should look like on a platform, and how they can win.

“We’re at an inflection point in social media,” says Oh. “We had one hypothesis with T2 and Pebble. It obviously didn’t pan out the way that we wanted it to.
But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.
</item 1>

<item 2>
Pittsburgh boy, 10, needs second liver transplant to save his life: ‘Only possible through love’ (FOX News - Health, 2023-12-09T16:31:40+00:00)
Dr. George Mazariegos, chief of pediatric transplantation at UPMC Children's Hospital of Pittsburgh, said Lucas is experiencing some of the more subtle changes
of liver failure.

PIG HEART RECIPIENT'S ‘LAST WISH’ WAS TO HELP DOCTORS LEARN FROM PROCEDURE, DYING MAN SAID: ‘WE WILL MISS HIM'

"Those include not being able to grow normally and not getting the proper levels of oxygen to his body and brain, because the blood is being shunted away from
the liver," he told Fox News Digital.

"Those are important and potentially life-threatening complications."

As of now, Lucas is still at home and "managing his symptoms," Mazeriegos said, but the doctor emphasized the importance of his receiving a timely transplant
before further complications occur.

next Image 1 of 2

prev Image 2 of 2

The Goellers are now seeking support again, launching the "Save Lucas" campaign across social media.
</item 2>

<item 3>
Pittsburgh boy, 10, needs second liver transplant to save his life: ‘Only possible through love’ (FOX News - Health, 2023-12-09T16:31:40+00:00)
God told me to start a Facebook page for him, and at the time I really wasn't into social media."

That page led to a viral national campaign — complete with local news stories and donated billboards — that attracted the attention of a Nebraska family.

ARKANSAS MILITARY VETERAN RECEIVES WORLD’S FIRST WHOLE-EYE AND PARTIAL-FACE TRANSPLANT

And in July of that year, at UPMC Children’s Hospital of Pittsburgh, little 2-year-old Lucas received the liver of 3-year-old Olivia Swedberg, who had passed
away a day earlier of brain cancer .

"I'm in awe when someone can take their suffering and turn it into something beautiful by donating and giving someone a second chance," said Goeller.

After that first transplant, her young son made a "miraculous recovery," Goeller said.
</item 3>

<item 4>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think Threads and others have encountered this as well,” adds Pebble co-founder and CTO Michael Greer, previously Discord engineering head. “The durability
of the network effect of Twitter is stronger than anticipated,” he says.

Despite these challenges, Pebble initially had decent retention. By week four, it was seeing retention rates of 30%, in fact. Its invite list was also working,
as 60% of people who received a Pebble invite converted to become users. In addition, some 10,000 users arrived at that list from early press, like TechCrunch’s
coverage of its first outside funding — a $1.1 million angel round that included investors like former Google VP Bradley Horowitz, Android co-founder Rich Miner
and the former CEO of Wikipedia, Katherine Maher. Data from Similarweb indicates the Pebble website topped 99,300 visits last month, indicating repeat usage.

The company’s hypothesis was that consumers were hungry for a Twitter alternative that prioritized trust, safety and moderation from the start.
</item 4>

<item 5>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think the competitive landscape evolved faster than we had thought,” he explains. “I didn’t think that quite as many people — established organizations and
newcomers — would try to do the same thing that we were doing and in very similar ways.”

Today, the market for Twitter alternatives is a crowded one, to say the least. There are numerous services for consumers to try, including the open source-based
platform Mastodon, a soon-to-be decentralized system from Bluesky, plus smaller startups like Spill, Spoutible and Post, as well as a new app from Meta called
Instagram Threads.

Then, of course, there’s the X factor to consider. The default platform for short posts retains some pull, despite — or perhaps because of — new owner Elon
Musk’s antics. Though the company may not have yet succeeded from a business and financial perspective, it’s been hard for others to duplicate its function as a
breaking news platform and place for spirited debates.
</item 5>

<item 6>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“We were really seized by people saying they wanted something that felt safer, where they could speak openly without dogpiling,” says Oh. “And so we were
confident that was an important thing to hit. Are there other things that are equally important and a great user experience? Yes.”

“Twitter is you have real people…saying real things,” Oh continues. “At the expense of using a word that’s overused, ‘authentic.’ And our theory was always if
you can create a space where people really feel like they can let down their hair, then you would get that authentic content. We didn’t quite get there,” she
admits.

Cselle agrees, noting that, perhaps, Pebble should have opened up enough space for disagreement to happen, while still drawing a hard line on the most
disagreeable parts of running a Twitter-like platform.

“I think we could have maybe moved a little bit further towards that,” he says. “There’s a lot of nice things that are said on Pebble, but maybe not enough
breaking news.
</item 6>

<item 7>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
Not enough conversation.”

There were other things that may have hurt Pebble’s traction, too, including its lack of a native mobile app. As T2, the startup prioritized building for the
web to be agile and more flexible, but lost out on the opportunities for discovery driven by the app stores. Notes Greer, “that might have been a mistake.”

In retrospect, Cselle also questions if the rebranding from T2 to Pebble may have been an issue.

“I’m wondering if the Pebble rebrand had actually softened the value prop[position] because T2 just encapsulated it so much — that it was really duplicative [of
Twitter],” he says.

Like most startup shutdowns, the answer as to “why” is not any one thing but a combination of factors. For Pebble, that’s certainly true.
</item 7>

<item 8>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
For Pebble, that’s certainly true. It was a perfect storm of competition, X’s continued traction, the lack of a native app, a brand that didn’t resonate and a
space that was maybe a little too safe to be as addictive and as fun as the original.

With Pebble’s winddown, being announced to users today at 9 AM PT, early adopters will have the option of exporting their Pebble archive as a zip file that
loads a mini web page showcasing all their old posts. Pebble won’t be directing users back to X however, or any other social network.

As for where the founders will now hang out?

“I don’t know. Maybe LinkedIn,” Cselle jokes.

Pebble is also returning a small amount of the funds left over to its investors, more as a courtesy to show financial responsibility. The Pebble website will be
shut down on November 1st and there aren’t further plans for the IP at this time.
</item 8>

<item 9>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
To that end, Pebble’s founding team included co-founder Sarah Oh, Twitter’s former human rights advisor.

Pebble continues to believe its approach to moderation was correct, even if it was not ultimately a factor in driving growth.

“We came in with a particular angle: kinder, safer. Trust and safety. And I think our approach to moderation that we did have on the site did work,” says
Cselle, adding that Pebble didn’t have some of the issues other federated platforms faced. For example, TechCrunch reported that Bluesky earlier this year had
strained its relationship with Black users, as it failed to crack down on bigotry on the site. Later, users on Bluesky were even creating usernames with racial
slurs, and yet the platform didn’t issue a public apology. Pebble never faced any such controversies.

But it may have positioned itself a little too far into the “kindness” territory, the founders now believe.
</item 9>

<item 10>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The grip Twitter, now called X, has on the market may be stronger than some believed. Unfortunately, that’s led to the first casualty among Twitter
alternatives, as the startup Pebble (formerly T2), is shutting down. The would-be X rival had grown a small but engaged community on its microblogging service
that aimed to dupe Twitter’s features, from its verification systems to functionality like DMs. But the company says they’ve run out of time to make Pebble
happen — the app maxed out at 3,000 daily active users, out of 20,000 registered users. That daily user figure fell to 1,000 daily users following its
rebranding from T2.

In part, the sizable competition coming from other Twitter rivals was to blame, says co-founder and CEO Gabor Cselle, who had exits to Google and Twitter prior
to starting Pebble.

“I think the competitive landscape evolved faster than we had thought,” he explains.
</item 10>
=== END ITEM id=c38c2aa7a91a4956 ===

=== ITEM id=5fdc7b5af7a90451 ===
Gold sentences and facts:
Sentence 0: Hope has been rising on Wall Street recently that the US economy is slowing from its recent hot pace by just the right amount.
  Fact 0.0: Hope has been rising on Wall Street recently.
  Fact 0.1: The hope on Wall Street is that the US economy is slowing from its recent hot pace.
  Fact 0.2: The hope on Wall Street is that the US economy is slowing by just the right amount.
Sentence 1: A shutdown would drag on the US economy, raising the risk of a recession, though financial markets have held up relatively well through past shutdowns.
  Fact 1.0: A shutdown would drag on the US economy.
  Fact 1.1: A shutdown would raise the risk of a recession.
  Fact 1.2: Financial markets have held up relatively well through past shutdowns.

Retrieved items:
<item 1>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Friday’s economic data shows that not only was inflation a touch cooler than expected in August, so was growth in spending by US consumers. That can be a
positive for inflation because it means not as many dollars are pouring into purchases. That in turn could give companies less encouragement to try to raise
prices further. But it may also dent what’s been a big driver keeping the US economy out of a recession.

“It came to a boil during a hot summer, and the temperature is really starting to come down,” said Brian Jacobsen, chief economist at Annex Wealth Management,
of spending growth by US consumers. “Higher energy prices, student loan debt repayments and real disposable incomes that have been on a declining trajectory
since June doesn’t bode well.” Oil prices have jumped to their highest level in more than a year, which is pressuring the economy by raising fuel costs for
everyone. A barrel of US crude sank 92¢ Friday to settle at $US90.79, but it’s still up sharply from $US70 in June.
</item 1>

<item 2>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Wall Street is also absorbing the ouster of Kevin McCarthy as the speaker of the House of Representatives. The unprecedented move to remove a speaker from the
position likely doesn’t change much in the short term, with funding for the US government set until November 17.

“That said, a leadership vacuum in the House raises the odds of a government shutdown when the current funding extension expires,” according to economists at
Goldman Sachs. A shutdown would drag on the US economy, raising the risk of a recession, though financial markets have held up relatively well through past
shutdowns. On Wall Street, big tech stocks helped to support the market after leading it lower a day earlier. They tend to move more sharply with expectations
for rates because high-growth stocks are seen as some of the biggest victims of high yields. In markets abroad, stock indexes were mixed across much of Europe.
</item 2>

<item 3>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
This is what the market has priced in today.”

The local gains came after most stocks slipped on Wall Street overnight, although the US market stayed near its highest level in 20 months following a mixed set
of reports that kept alive questions about whether the world’s largest economy can pull off a perfect landing where it kills high inflation, but avoids a
recession. The S&P 500 edged down by 0.1 per cent for its first back-to-back loss since October. The Dow Jones slipped 0.2 per cent and the Nasdaq composite
rose 0.3 per cent. Stocks were down more sharply in Asia amid worries about the health of China’s economy, the world’s second largest. US stocks and Treasury
yields wavered after reports showed that US employers advertised far fewer jobs at the end of October than expected, while growth for services businesses
accelerated more last month than expected. Hope has been rising on Wall Street recently that the US economy is slowing from its recent hot pace by just the
right amount.
</item 3>

<item 4>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.
</item 4>

<item 5>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
At Expedia Group, another stronger-than-forecast report sent its stock nearly 22 per cent higher for its best week since the market was surging out of the
coronavirus crash in early 2020. But it was interest rates, yields and inflation that were at the centre of all the wild movements for financial markets around
the world.

Before last week, stocks had been struggling under the weight of rapidly rising Treasury yields. Those yields were in turn catching up to the Fed’s main
interest rate, which is above 5.25 per cent and at its highest level since 2001. Loading Higher rates and yields slow the economy, hurt prices for investments
and raise the risk of something breaking within the financial system, such as the three high-profile US bank failures that rattled financial markets during the
spring. “It was really fear that the Fed was going to go too far,” said Katie Nixon, chief investment officer for Northern Trust Wealth Management. The Fed put
such pressure on the economy intentionally, hoping to starve inflation of its fuel.
</item 5>

<item 6>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
That growing understanding has sent yields in the bond market to their highest levels in more than a decade, which in turn has undercut prices for stocks and
other investments. Treasury yields rose again on Tuesday following a mixed batch of reports on the economy. The yield on the 10-year Treasury edged up to 4.55
per cent from 4.54 per cent late on Monday and is near its highest level since 2007. It’s up sharply from about 3.5 per cent in May and from 0.5 per cent about
three years ago. One economic report on Tuesday showed confidence among consumers was weaker than economists expected. That’s concerning because strong spending
by US households has been a bulwark keeping the economy out of a long-predicted recession.

Besides high interest rates, a long list of other worries is also tugging at Wall Street. The most immediate is the threat of another US government shutdown as
Capitol Hill threatens a stalemate that could shut off federal services across the country.
</item 6>

<item 7>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 7>

<item 8>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Loading Wall Street has dealt with such shutdowns in the past, and stocks have historically been turbulent in the run-up to them, according to Lori Calvasina,
strategist at RBC Capital Markets. After looking at the seven shutdowns that lasted 10 days or more since the 1970s, she found the S&P 500 dropped an average of
roughly 10 per cent in the three months heading into them. But stocks managed to hold up rather well during the shutdowns, falling an average of just 0.3 per
cent, before rebounding meaningfully afterward. Wall Street is also contending with higher oil prices, shaky economies around the world, a strike by US
autoworkers that could put more upward pressure on inflation and a resumption of US student-loan repayments that could dent spending by households.

On Wall Street, the vast majority of stocks fell under such pressures, including 90 per cent of those within the S&P 500.
</item 8>

<item 9>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
“Considering the damage overnight in the US, Australian markets held on pretty well,” he said. “I thought it would be a bigger down day across the board.”
Glover said the market was volatile but quite range-bound, similar to previous periods of high inflation in the 1940s and 1970s. Elsewhere, Wall Street’s ugly
September got even worse on Tuesday, as a sharp drop for stocks brought them back to where they were in June. The S&P 500 tumbled 1.5 per cent for its fifth
loss in the last six days. The Dow Jones dropped 1.1 per cent, and the Nasdaq composite lost 1.6 per cent.

Loading September has brought a loss of 5.2 per cent so far for the S&P 500, putting it on track to be the worst month of the year by far, as the realisation
sets in that the Federal Reserve will indeed keep interest rates high for a long time.
</item 9>

<item 10>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Fed put such pressure on the economy intentionally, hoping to starve inflation of its fuel. It wants the job market to cool, particularly pay raises going
to workers. The Fed fears too-strong pay gains could create a vicious cycle that keeps inflation high.

Analysts said Friday’s jobs report offered encouraging signals for the Fed, with average hourly earnings rising less in October from September than expected,
though it doesn’t mean the job is done. Treasury yields in the bond market tumbled immediately after the jobs report, releasing more of the pressure that had
built up on Wall Street. The yield on the 10-year Treasury eased to 4.52 per cent from 4.67 per cent late Thursday and from more than five per cent last week,
when it hit its highest level since 2007. In the US, traders are moving up expectations for when the first cut to interest rates by the Fed could happen,
potentially by the summer, according to data from CME Group. Such cuts can act like steroids for financial markets.
</item 10>
=== END ITEM id=5fdc7b5af7a90451 ===

=== ITEM id=e016bedf6d80b17c ===
Gold sentences and facts:
Sentence 0: Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.
  Fact 0.0: Al Saadi said her frustration with Meta is common among Palestinians.
  Fact 0.1: Al Saadi said her frustration with Meta is common among Palestinians in occupied territories.
  Fact 0.2: Al Saadi said her frustration with Meta is common among Palestinians across the diaspora.
Sentence 1: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents.
  Fact 1.0: Meta wants to shift the burden of monitoring teens' social media usage back to the app stores.
  Fact 1.1: Meta wants to shift the burden of monitoring teens' social media usage back to parents.

Retrieved items:
<item 1>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.
</item 1>

<item 2>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.
</item 2>

<item 3>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
His argument essentially was that the improvements and safety features Meta was implementing were not enough and not effective.

However, a policy position like the one Meta announced today is not something that would have been rushed out in a matter of days as a PR response to Bejar’s
claims, but something Meta has been stewing on for some time. After all, the app stores already have age guidelines, so it would seem to follow that they should
also enforce them, if enforcement is to be regulated — or that seems to be Meta’s thinking.

If anything, the call to bring the app stores into the fight is a response to the lack of action in Congress to pass online child safety regulations, like the
hotly debated KOSA (the Kids Online Safety Act) leaving U.S. states to individually create laws due to a void of federal guidelines.

Utah, for instance, passed a measure that will require social networks to give Utah parents access to their children’s posts, messages and comments, and would
block social media access during certain hours.
</item 3>

<item 4>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Most recently, Instagram users accused Meta of “shadowbanning” their Stories about Palestine.

It’s the latest in a lengthy history of incidents on Meta platforms that reflect an inherent bias against Palestinian users in its processes, as documented by
years of complaints from both inside and outside the company. The company may not intentionally suppress specific communities, but its moderation practices
often disproportionately affect Palestinian users.

For instance, Meta struggles to navigate the cultural and linguistic nuances of Arabic, a language with over 25 dialects, and has been criticized for neglecting
to adequately diversify its language resources. The company’s black-and-white policies often preclude it from effectively moderating any nuanced topic, like
content that discusses violence without condoning it. Advocacy groups have also raised concerns that Meta’s partnerships with government agencies, such as the
Israeli Cyber Unit, politically influence the platform’s policy decisions.
</item 4>

<item 5>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.
</item 5>

<item 6>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“And so I think that is what is the most problematic right now about that particular rule, and the way that it’s enforced.”

Over the 13 days leading up to the ceasefire between Hamas and Israel, 7amleh documented more than 500 reports of Palestinian “digital rights violations,”
including the removal and restriction of content, hashtags and accounts related to the conflict.

Meta blamed some of the instances of perceived censorship to technical issues, like one that prevented users in Palestine and Colombia from posting Instagram
Stories. It attributed others to human error, like blocking the hashtag for Al-Aqsa Mosque, the holy site where Israeli police clashed with Ramadan worshippers,
because it was mistaken for a terrorist organization. The company also blocked journalists in Gaza from WhatsApp without explanation.

The same month, a group of Facebook employees filed internal complaints accusing the company of bias against Arab and Muslim users. In internal posts obtained
by BuzzFeed News, an employee attributed the bias to “years and years of implementing policies that just don’t scale globally.”
</item 6>

<item 7>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Although harassment and abuse are still rampant on Instagram and Facebook, he said, the company has been responsive to suspending accounts with patterns of
targeting other users.

The company has also made more contact with regional Palestinian organizations since 2021, York added, but it’s been slow to implement recommendations from EFF
and other advocacy groups. It’s “very clear” that Meta is not putting the same resources behind Arabic and other non-English languages, York said, compared to
the attention Meta gives to countries that have the most regulatory pressure. Moderation of English and other European languages tends to be more comprehensive,
for example, because the EU enforces the Digital Services Act.

In Meta’s response to the report, Miranda Sissons, the company’s director of human rights, said that Meta was “assessing the feasibility” of reviewing Arabic
content by dialect. Sissons said that the company has “large and diverse teams” who understand “local cultural context across the region,” including in
Palestine.
</item 7>

<item 8>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
“U.S. states are passing a patchwork of different laws, many of which require teens (of varying ages) to get their parent’s approval to use certain apps, and
for everyone to verify their age to access them,” she wrote. “Teens move interchangeably between many websites and apps, and social media laws that hold
different platforms to different standards in different states will mean teens are inconsistently protected,” Davis said.

Meta’s proposed “better way” of requiring parents to manage a teen’s app downloads pushes the burden back to the app stores and to parents, who often today do
not monitor or manage their teens’ smartphone use. But while it’s true that the app stores and smartphone makers would know the ages of their users, requiring
parents to approve apps is the equivalent of overriding parental discretion and choice.

Other entertainment ratings — like those for TV, movies or video games — are meant to be guidelines for parents, not laws. Meta’s position that apps are somehow
different, then, is a fairly extreme position to take.
</item 8>

<item 9>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“It feeds into the bias against Palestinian narrative genuinely. It really takes the balance against Palestinians in a situation where there’s a huge asymmetry
of power.”

A history of suppression

Content about Palestine is disproportionately scrutinized, as demonstrated during the last severe violent outbreak between Hamas and Israel two years ago. Amid
the violence following the May 2021 court ruling to evict Palestinian families from Sheikh Jarrah, a neighborhood in occupied East Jerusalem, users across
Facebook and Instagram accused Meta of taking down posts and suspending accounts that voiced support for Palestinians.

The digital rights nonprofit Electronic Frontier Foundation (EFF) described Meta’s actions in 2021 as “systemic censorship of Palestinian voices.” In its 2022
report of Palestinian digital rights, 7amleh said that Meta is “still the most restricting company” compared to other social media giants in the extent of its
moderation of the Palestinian digital space.
</item 9>

<item 10>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
During the last violent outbreak between Hamas and Israel in 2021, a report commissioned by Meta and conducted by a third party concluded that the company’s
actions had an “adverse human rights impact” on Palestinian users’ right to freedom of expression and political participation.

Meta communications director Andy Stone declined to comment on the record, and pointed TechCrunch to Meta’s newsroom post about the conflict, which was updated
on Oct. 18 with a statement addressing the shadowbanning claims.

“Our policies are designed to keep people safe on our apps while giving everyone a voice,” the post stated. “We apply these policies equally around the world
and there is no truth to the suggestion that we are deliberately suppressing voice … We can make errors and that is why there is an appeals process for people
to tell us when they think we have made the wrong decision, so we can look into it.”

The belief that Meta shadowbans, or limits the visibility of, content about Palestine is not new.
</item 10>
=== END ITEM id=e016bedf6d80b17c ===

=== ITEM id=11b6876b646f7be2 ===
Gold sentences and facts:
Sentence 0: A number of videos posted to X since the attacks have been identified as entirely unrelated to the conflict — including footage that was filmed last month in Egypt and even a clip from a video game that had been posted to the platform with a (false) claim it showed Hamas missile attacks on Israel.
  Fact 0.0: A number of videos posted to X since the attacks have been identified as entirely unrelated to the conflict.
  Fact 0.1: One unrelated video was filmed last month in Egypt.
  Fact 0.2: Another unrelated item was a clip from a video game.
  Fact 0.3: The video game clip was posted to X with a false claim that it showed Hamas missile attacks on Israel.
Sentence 1: We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.
  Fact 1.0: The outlet reached out to the Commission with questions.
  Fact 1.1: The questions ask how the Commission plans to respond to this latest analysis of X's role in spreading disinformation.

Retrieved items:
<item 1>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 1>

<item 2>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
They must also clearly communicate their T&Cs to users and be able to demonstrate proper enforcement.

Larger platforms like X also have additional obligations to identify and mitigate systemic risks such as disinformation, gender-based violence or negative
effects on the exercise of fundamental rights.

Additionally, the regulation includes a “crisis response” mechanism (Article 36) which enables the Commission to adopt rapid-fire measures on larger platforms
in situations of “serious threat”, such as war.

Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show
Hamas missile attacks on Israel but was actually footage from a video game.

Meanwhile X’s ability to respond internally to reports of content problems have been drastically pared back after Musk’s takeover last year — which saw major
layoffs, including in content moderation and human rights as part of the new billionaire owner’s bid to improve the platform’s profitability.
</item 2>

<item 3>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 3>

<item 4>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
In a very swift test of the European Union’s newly updated content moderation rulebook, the bloc has fired a public warning at Elon Musk–owned X (formerly
Twitter) for failing to tackle illegal content circulating on the platform in the wake of Saturday’s deadly attacks on Israel by Hamas terrorists based in the
Gaza Strip.

The European Commission has also raised concerns about the spread of disinformation on X related to the terrorist attacks and their aftermath.

Unlike terrorism content, disinformation is not illegal in the EU per se. However, the EU’s Digital Services Act (DSA) puts an obligation on X — as a so-called
very large online platform — to mitigate risks attached to harmful falsehoods as well as to act diligently on reports of illegal content.

Following the terrorist attacks by Hamas against 🇮🇱, we have indications of X/Twitter being used to disseminate illegal content & disinformation in the EU.
</item 4>

<item 5>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
It replied with an automated email reading: “Busy now, please check back later.”

Update: X’s global government affairs account has posted a thread responding to the EU’s criticism in which it claims the data shows other social networking
services saw greater changes in subscriber growth and total growth for pro-Kremlin accounts, and for engagement and volume; it also writes that: “More than 700
unique Community Notes have appeared on posts related to the Ukraine conflict. Through Community Notes’ new ‘Notes on Media’ feature, which automatically
matches notes on an image or video to other posts that contain matching media, these notes appeared on over 1,200 additional posts. These notes are in numerous
languages and are written specifically for local audiences.”

Back in April, X/Twitter was named as one of 19 larger platforms that face the strictest level of regulation under the EU’s DSA.
</item 5>

<item 6>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“In this context, 832 videos related to the war have been fact-checked, of which 211 have been removed,” Jourova noted.

The EU also flagged reporting by Microsoft that told it Bing Search had either promoted information or downgraded questionable information in relation to almost
800,000 search queries related to the Ukraine crisis.

Jourova’s speech also highlighted a couple of other areas where she urged Code signatories to go further — calling (yet again) for more consistent moderation
and investment in fact-checking, especially in smaller Member States and languages.

She also criticized platforms over access to data, saying they must step up efforts to make sure researchers are empowered to scrutinize disinformation flows
“and contribute to the necessary transparency”.

Both are areas where X/Twitter under new owner, Elon Musk, has moved out of step with EU expectations on countering disinformation.

Twitter (now X) was an original signatory to the disinformation Code but Musk took the platform out of the initiative back in May, as critical scrutiny of his
actions dialled up in the EU.
</item 6>

<item 7>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
The law puts an obligation on so-called very large online platforms (VLOPs) to assess societal risks attached to their use of algorithms — and put in place
“reasonable, proportionate and effective mitigation measures” for identified risks, with their reporting and mitigation plans subject to independent audit and
oversight by the Commission.

We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.

Update: A Commission spokesperson told us:

While the Code is voluntary, compliance with the DSA is not. The DSA is now legally binding on X/Twitter. This includes in particular the obligation of Very
Large Online Platforms, such as X/Twitter to mitigate the risks their platforms pose for the spread of disinformation. The TrustLab study to measure two
Structural Indicators covers signatories of the Code at the time of the launch of the exercise, which at the time included X/Twitter.
</item 7>

<item 8>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
X (formerly Twitter) has been called out in the European Union for having the worst ratio of disinformation/misinformation to posts not spreading falsehoods
among mainstream social networks.

In a speech today discussing the latest updates from other platforms that have signed up to combat disinformation under a beefed up EU Code of Practice that’s
been linked to compliance with the the bloc’s legally binding Digital Services Act (DSA), it commissioner for values and transparency, Vera Jourova, said X’s
poor performance was assessed during a pilot phase of testing of new methodology developed by Code signatories.

“A big novelty under the Code is now the publication of the first set of structural indicators such as how easy it is to find disinformation content, how much
engagement such content receives or indicators about sources. These indicators — developed by the signatories — constitute an unprecedented and novel insight
into disinformation on online platforms,” she suggested. “Such insight is crucial to understand how effective platforms efforts are to fight this threat even
more efficiently.
</item 8>

<item 9>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
On Wednesday, in an “urgent” letter to Musk, the EU’s internal market commissioner Thierry Breton said the bloc had seen “indications” from “qualified sources”
that X is being used to disseminate illegal content and disinformation in the EU” following Saturday’s attacks — before reminding X of the DSA’s “very precise
obligations” vis-à-vis content moderation.

“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” Breton also warned then.

In its latest press release, the EU said it has sent X a formal request for information under the DSA.

“This request follows indications received by the Commission services of the alleged spreading of illegal content and disinformation, in particular the
spreading of terrorist and violent content and hate speech,” it wrote, adding that the ask also addresses other aspects of DSA compliance.

The pan-EU regulation puts a series of governance obligations on digital services and platforms with the aim of ensuring companies are responding to reports of
illegal content.
</item 9>

<item 10>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In this case, failure to reply by the deadline could lead to the imposition of period penalties,” the EU added.

Earlier this year, Musk pulled out of the EU’s Code of Practice on online disinformation. In a response to X leaving the voluntary Code, Breton warned:
“Obligations remain. You can run but you can’t hide.”

In recent days, TikTok and Meta have also been warned by the EU over disinformation related to the Israel-Hamas war. But back in September, a study commissioned
via the EU’s Code of Practice suggested X is the worst of the major platforms when it comes to spreading disinformation.

This report was substantially revised to clarify a number of aspects, including — most saliently — that the EU has sent a formal request for information to X at
this point. This could prefigure a formal announcement of an investigation but is not, yet, that formal technical step.
</item 10>
=== END ITEM id=11b6876b646f7be2 ===
