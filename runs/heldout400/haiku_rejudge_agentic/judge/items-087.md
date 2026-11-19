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

=== ITEM id=cb8669a864c06534 ===
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
The grip Twitter, now called X, has on the market may be stronger than some believed. Unfortunately, that’s led to the first casualty among Twitter
alternatives, as the startup Pebble (formerly T2), is shutting down. The would-be X rival had grown a small but engaged community on its microblogging service
that aimed to dupe Twitter’s features, from its verification systems to functionality like DMs. But the company says they’ve run out of time to make Pebble
happen — the app maxed out at 3,000 daily active users, out of 20,000 registered users. That daily user figure fell to 1,000 daily users following its
rebranding from T2.

In part, the sizable competition coming from other Twitter rivals was to blame, says co-founder and CEO Gabor Cselle, who had exits to Google and Twitter prior
to starting Pebble.

“I think the competitive landscape evolved faster than we had thought,” he explains.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, X, Google, Twitter, Pebble, Gabor Cselle, Pebble, the Twitter alternative previously known as T2, is shutting down
</item 1>

<item 2>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think Threads and others have encountered this as well,” adds Pebble co-founder and CTO Michael Greer, previously Discord engineering head. “The durability
of the network effect of Twitter is stronger than anticipated,” he says.

Despite these challenges, Pebble initially had decent retention. By week four, it was seeing retention rates of 30%, in fact. Its invite list was also working,
as 60% of people who received a Pebble invite converted to become users. In addition, some 10,000 users arrived at that list from early press, like TechCrunch’s
coverage of its first outside funding — a $1.1 million angel round that included investors like former Google VP Bradley Horowitz, Android co-founder Rich Miner
and the former CEO of Wikipedia, Katherine Maher. Data from Similarweb indicates the Pebble website topped 99,300 visits last month, indicating repeat usage.

The company’s hypothesis was that consumers were hungry for a Twitter alternative that prioritized trust, safety and moderation from the start.

Graph facts:
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> moderation
- Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design -MENTIONS-> moderation
- Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society -MENTIONS-> moderation
- CEO David Baszucki’s mission to make Roblox a billion-player platform -MENTIONS-> moderation
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> moderation

Entities: moderation, Android, Threads, TechCrunch, Similarweb, Katherine Maher, Google, safety, trust, Discord, Pebble, Pebble, the Twitter alternative
previously known as T2, is shutting down, Twitter, Michael Greer, $1.1 million angel round
</item 2>

<item 3>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
For Pebble, that’s certainly true. It was a perfect storm of competition, X’s continued traction, the lack of a native app, a brand that didn’t resonate and a
space that was maybe a little too safe to be as addictive and as fun as the original.

With Pebble’s winddown, being announced to users today at 9 AM PT, early adopters will have the option of exporting their Pebble archive as a zip file that
loads a mini web page showcasing all their old posts. Pebble won’t be directing users back to X however, or any other social network.

As for where the founders will now hang out?

“I don’t know. Maybe LinkedIn,” Cselle jokes.

Pebble is also returning a small amount of the funds left over to its investors, more as a courtesy to show financial responsibility. The Pebble website will be
shut down on November 1st and there aren’t further plans for the IP at this time.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> LinkedIn
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> LinkedIn
- Here’s how Rainforest, a budding Stripe rival, aims to win over software companies -MENTIONS-> LinkedIn
- Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer -MENTIONS-> LinkedIn
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> LinkedIn

Entities: LinkedIn, TechCrunch, X, Pebble, Pebble, the Twitter alternative previously known as T2, is shutting down, Cselle, Pebble website shutdown
</item 3>

<item 4>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
The founders don’t regret their experience building Pebble, even though it didn’t turn out as they hoped, they said.

Says Cselle, “One thing that I learned is that there’s absolutely an audience that wants to see a new kind of Twitter-like platform built and will ask for the
features that Twitter has.” He suspects the team might stay together to work on something new after Pebble wraps, taking the learnings from Pebble with them.
That includes better communicating to users what great content should look like on a platform, and how they can win.

“We’re at an inflection point in social media,” says Oh. “We had one hypothesis with T2 and Pebble. It obviously didn’t pan out the way that we wanted it to.
But I think we’ll look back and see this past year is a really important turning point for the role that social media plays in our lives,” she adds.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Pebble, Pebble, the Twitter alternative previously known as T2, is shutting down, Twitter, Cselle, Oh, social media
</item 4>

<item 5>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
To that end, Pebble’s founding team included co-founder Sarah Oh, Twitter’s former human rights advisor.

Pebble continues to believe its approach to moderation was correct, even if it was not ultimately a factor in driving growth.

“We came in with a particular angle: kinder, safer. Trust and safety. And I think our approach to moderation that we did have on the site did work,” says
Cselle, adding that Pebble didn’t have some of the issues other federated platforms faced. For example, TechCrunch reported that Bluesky earlier this year had
strained its relationship with Black users, as it failed to crack down on bigotry on the site. Later, users on Bluesky were even creating usernames with racial
slurs, and yet the platform didn’t issue a public apology. Pebble never faced any such controversies.

But it may have positioned itself a little too far into the “kindness” territory, the founders now believe.

Graph facts:
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> moderation
- Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design -MENTIONS-> moderation
- Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society -MENTIONS-> moderation
- CEO David Baszucki’s mission to make Roblox a billion-player platform -MENTIONS-> moderation
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> moderation

Entities: moderation, trust and safety, TechCrunch, racial slurs, Twitter, Bluesky, Pebble, Pebble, the Twitter alternative previously known as T2, is shutting
down, Sarah Oh, Black users, bigotry, Cselle
</item 5>

<item 6>
Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne (TechCrunch, 2023-11-07T17:08:04+00:00)
Another app that spun out from laid-off former Twitter employees, Pebble (formerly T2), recently shut down after a year, growing to just 20,000 registered
users. Its CEO, Gabor Cselle, previously had exits to Google and Twitter, but accepted that Pebble simply wasn’t competitive enough to stay afloat. But while
Pebble sought to essentially clone Twitter, Spill’s team is building something that looks very little like it.

“People are looking for something new,” Terrell said. “I think things that have really defined, unique value propositions are going to win over the long term —
it might not be like there’s one winner-take-all.”

Earlier this year, TechCrunch covered the founding of Spill, piquing the interest of Kapor Capital, which ended up writing the company its first check.

“We had over 25,000 people join our waiting list in less than 24 hours from that article,” Terrell said.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Google, Kerry Washington, Spill, Kapor Capital, Terrell, Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne,
Pebble, Gabor Cselle, Twitter
</item 6>

<item 7>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
“I think the competitive landscape evolved faster than we had thought,” he explains. “I didn’t think that quite as many people — established organizations and
newcomers — would try to do the same thing that we were doing and in very similar ways.”

Today, the market for Twitter alternatives is a crowded one, to say the least. There are numerous services for consumers to try, including the open source-based
platform Mastodon, a soon-to-be decentralized system from Bluesky, plus smaller startups like Spill, Spoutible and Post, as well as a new app from Meta called
Instagram Threads.

Then, of course, there’s the X factor to consider. The default platform for short posts retains some pull, despite — or perhaps because of — new owner Elon
Musk’s antics. Though the company may not have yet succeeded from a business and financial perspective, it’s been hard for others to duplicate its function as a
breaking news platform and place for spirited debates.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, X, Elon Musk, Spill, Mastodon, Instagram Threads, Bluesky, Pebble, Spoutible, Post, Twitter alternatives, Pebble, the Twitter
alternative previously known as T2, is shutting down, Twitter
</item 7>

<item 8>
Pebble, the Twitter alternative previously known as T2, is shutting down (TechCrunch, 2023-10-24T16:00:41+00:00)
Not enough conversation.”

There were other things that may have hurt Pebble’s traction, too, including its lack of a native mobile app. As T2, the startup prioritized building for the
web to be agile and more flexible, but lost out on the opportunities for discovery driven by the app stores. Notes Greer, “that might have been a mistake.”

In retrospect, Cselle also questions if the rebranding from T2 to Pebble may have been an issue.

“I’m wondering if the Pebble rebrand had actually softened the value prop[position] because T2 just encapsulated it so much — that it was really duplicative [of
Twitter],” he says.

Like most startup shutdowns, the answer as to “why” is not any one thing but a combination of factors. For Pebble, that’s certainly true.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Pebble, Pebble, the Twitter alternative previously known as T2, is shutting down, Twitter, Greer, Cselle, Pebble shutting down, lack of a
native mobile app, rebranding from T2 to Pebble
</item 8>

<item 9>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> SBF Trial
- Founders, are events useful? -MENTIONS-> Pixel 8
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Pixel 8
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Google’s hardware event

Entities: ad-free tier, Pixel 8, IronNet, Keith Alexander, NSA, going public, blockchain, Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok
tests an ad-free tier, Virginia, TechCrunch, X, startups, Google, TikTok, cybersecurity
</item 9>

<item 10>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Pebble, Pebble, the Twitter alternative previously known as T2, is shutting down, Twitter, Cselle, Oh
</item 10>
=== END ITEM id=cb8669a864c06534 ===

=== ITEM id=3f4b6538f45aebfb ===
Gold sentences and facts:
Sentence 0: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 0.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.
  Fact 0.1: The three categories Google's public documentation names for ranking local results are relevance, distance, and prominence.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: A woman admits she may have misinterpreted the evidence.
  Fact 1.1: The woman maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google "siphons off" news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google "siphons off" news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, The Age, Google
</item 1>

<item 2>
Graph query row:
first_article: Is Google Search better than the rest? And is that fair?
first_published_on: 2023-10-22T18:00:00+00:00
second_article: When it comes to generative AI in the enterprise, CIOs are taking it slow
second_published_on: 2023-12-15T18:45:09+00:00
connecting_entities: ['Microsoft', 'generative AI']
first_report_claims_and_responses: ['Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)\nThe process of this test did
remind me how much data Google collects — not just my searches but anything I say to a smart speaker, any search result or ad that I click — which will be a
point in favour of DuckDuckGo for some, even if it didn’t perform as well and is harder to read. The data collection does also mean that Google (not just search
but also mobile widgets and other services) personalises to me more effectively. Loading Bing has overhauled its layout over the past year to be very graphic
and augmented by generative AI explanations from its chatbot. I like the idea of being able to ask follow-up questions in searches this way, but in practice
it’s not really there yet. In a search for “kids BMX bike green” (Google did pretty well and DuckDuckGo ignored the green part), Bing Chat offered a comparison
of a few different options but they were too expensive. I asked, “any under $200?” and it replied with more options that were all over $300.', 'Is Google Search
better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)\nand it replied with more options that were all over $300. Another observation from
this test is that I tend to use the Google Search bar as much for returning to familiar sites as I do for searching for information, which means switching to
another provider could be temporarily annoying simply because it wouldn’t remember all the places I’ve been. Is Google as the default unfair?\n\nIn a response
to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and device makers
simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search provider was
easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want to,” he said.
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”', 'Is Google Search better than the rest? And is that
fair? (The Age, 2023-10-22T18:00:00+00:00)\n“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also
points out that, when faced with a situation where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made
Yahoo! the default on Firefox, most users changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the
Microsoft default — “Google” was the number one search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point,
and many of us will be well accustomed to dodging all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC.
But all of these arguments do ignore the fact that Google and its competitors are not on an even playing field. The question remains, why do users prefer
Google?', 'Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)\nThe question remains, why do users prefer Google? And
if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down to its scale, its longevity, its ubiquity and all the
user data it sucks up, which no other company could possibly match.\n\nFor the US, which has to prove that Google somehow broke the law to achieve the status
quo, this legal case could be an uphill battle. But for the rest of us, it offers a rare opportunity for some insight into what the tech giant does with its
enormous market share. Is it still focused on being the best? Or, as has been alleged, does it use its ubiquity to squeeze us for money even at the expense of
product quality? How ‘organic’ are Google’s results? This month, Wired posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that
Google had accidentally revealed during the trial that it manipulates people’s search queries for maximum ad revenue.', 'Is Google Search better than the rest?
And is that fair? (The Age, 2023-10-22T18:00:00+00:00)\nThe example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”.
Loading Google rejected this in very strong terms, saying the piece was misleading and inaccurate while denying ever altering search terms. Wired removed the
article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing
people are to accept foul play on Google’s part. A suspicion of privacy invasion and inappropriate data handling follows the company at all times.\n\nGrey
herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may have
misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s official
search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not sponsored.', 'Is
Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)\nthe list of blue links in search results that are not sponsored. “Ad
keyword matching is a long-standing and well-known process that is designed to connect people to relevant ads. A separate process, which has nothing to do with
ads, is used to match organic results to a query,” he said. “It’s no secret that Google Search looks beyond the specific words in a query to better understand
their meaning ... If you make a spelling mistake, or search for a term that’s not on a page but where the page has a close synonym, or if you aren’t even sure
exactly how to search for something, our meaning matching systems help.” It is true that some search queries result in more ads than others, and what ads you
get can vary wildly depending on your request, which is the result of the keyword matching Google uses to sell ads. But Google denies pushing users towards the
most monetisable subjects.', 'Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)\nBut Google denies pushing users
towards the most monetisable subjects.\n\nSimilarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was
a lot better 10 years ago”, which could absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when
there’s a much simpler explanation. In 2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which
Google had to index constantly, whereas in 2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live
information that may be nested within various services and apps. What really happens when we search? Google has published extensive information about its Search
procedures, so it’s not difficult to get its own take on what happens in the background when you send off a query.', 'Is Google Search better than the rest? And
is that fair? (The Age, 2023-10-22T18:00:00+00:00)\nGoogle’s software is constantly crawling the web and updating its central index, which itself is more than
100 million gigabytes in size. When you enter a query, Search uses a few quick processes to decide the meaning of your question, including what kind of
information you might be after. Then, beyond just looking for sites with the words you typed in, the system uses machine learning to rank results according to
what it thinks you’re looking for, prioritising high quality and easy to use sites. If you allow it, Google does bring in some information it knows about you to
further refine results, including your location and web history. If you’re looking to turn this off, it’s in your Google account settings called “Web & App
Activity”. Google says it does not infer personal information like race or religion to shape results. Loading Google works with independent “search quality
raters” to gauge the effectiveness of its processes, and says it ran almost 900,000 quality tests in 2022.', 'Is Google Search better than the rest? And is that
fair? (The Age, 2023-10-22T18:00:00+00:00)\nSome critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise
on purpose, which widens the pool of content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still
returning results that were justifiably related to the input.']
second_report_claims_and_responses: ['When it comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch, 2023-12-15T18:45:09+00:00)\nTo hear
the hype from vendors, you would think that enterprise buyers are all in when it comes to generative AI. But like any newer technology, large companies tend to
move cautiously. Throughout this year, as vendors feverishly announced new generative AI-fueled products, CIOs took note.\n\nSome companies have actually been
looking to cut back on spending, or at least stay even, not necessarily looking for new ways to spend money. The big exception is when technology enables
companies to operate more efficiently, and do more with less.\n\nGenerative AI certainly has the potential to do that, but it also has its own costs associated
with it, whether it’s a higher cost for these features in a SaaS product or the price for hitting a large language model API if you’re building your own
software internally.\n\nEither way, it’s important for the folks implementing the technology to understand if they are getting a return on their investment.',
'When it comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch, 2023-12-15T18:45:09+00:00)\nA July Morgan Stanley survey of large
company CIOs found that many were proceeding cautiously, with 56% of respondents reporting that generative AI was having an impact on their investment
priorities, but only 4% had actually launched significant projects. In fact, most were still in the evaluation or proof of concept phase. This may be a fast
moving area, but it fits with what we’re hearing in conversations with CIOs as well.\n\nThat said, much like the consumerization of IT a decade ago, CIOs are
under pressure to deliver the kind of experiences people are seeing when they play with ChatGPT online, says Jon Turow, a partner at Madrona Ventures.\n\n“I
think it’s undeniable that enterprise employees, who are the internal customers of the CIO or CTO, have all tried ChatGPT and they know what amazing looks like.
They know where it’s early, and they know where it’s inspiring, and for lack of a better word, where they see greatness.', 'When it comes to generative AI in
the enterprise, CIOs are taking it slow (TechCrunch, 2023-12-15T18:45:09+00:00)\nAnd so CIOs are under pressure to deliver that level,” Turow told
TechCrunch.\n\nIt has created a tension between this desire to please the internal customers, especially when some of that pressure could be coming from the
CEO, and a CIO’s natural tendency to move cautiously, even with something as potentially transformative as generative AI. That’s going to take setting up some
structure and organization around how this gets implemented over time, says Jim Rowan, principal at Deloitte, who is working with clients around how to build
generative AI across companies in an organized fashion.\n\n“A lot of the way we’re working with companies is thinking about what is the infrastructure that they
need to be successful. By infrastructure, I don’t necessarily mean technology, but who are the people, what are the processes and the governance…and giving them
the capabilities to set that up,” Rowan said. A big part of that is talking about use cases and how to use the technology to address a given problem.', 'When it
comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch, 2023-12-15T18:45:09+00:00)\nThis is in line with how CIOs we spoke to are
approaching implementing this in their organizations. Monica Caldas, CIO at insurance company Liberty Mutual, started with a few-thousand-person proof of
concept, and is looking for ways to expand that for her 45,000 employee company.\n\n“We know generative AI will continue to play a critical role in virtually
every part of our company, so we’re investing in many use cases to further develop and refine them in service of supporting our employees and giving them better
internal capabilities,” she said.\n\nMike Haney, CIO at Battelle, a firm focused on science and technology, has also been exploring generative AI use cases this
year. “So we’ve been doing this whole push for AI over the last maybe six or nine months and we’re at the point right now where we’re building specific use
cases for each different team and function within the firm.”', 'When it comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch,
2023-12-15T18:45:09+00:00)\nHe cautions that it’s early, and they are still exploring ways in which it can help, but so far the results have been good in terms
of offering more efficient ways to do things.\n\nKathy Kay, executive VP and CIO at Principal Financial Group, a financial services company, says her company
started from scratch with a study group. “So any employees who had an interest or passion, we allowed them to join so there’s about 100 people. It’s a
combination of engineers and business people, and we are curating probably 25 use cases now that they’ve gone through, and three will be going into production
[soon],” she said.', 'When it comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch, 2023-12-15T18:45:09+00:00)\nSharon Mandell, CIO at
Juniper Networks, says that her company is participating in an initial pilot with Microsoft around Copilot for Office 365, and anecdotally, she has heard a
range of feedback from people who love it to those who are less impressed, but she says trying to measure increased productivity remains a challenge, even with
Microsoft beginning to provide dashboards that at least show the level of adoption and usage.\n\n“The hard thing about this is you don’t have data on people’s
level of productivity. So no matter what, you’re using somewhat anecdotal information until you get really good at understanding these dashboards from Microsoft
showing you how people are using it,” she said.', 'When it comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch,
2023-12-15T18:45:09+00:00)\nAs companies hear about the potential power of generative AI, it’s only natural that they would want to learn more about it and put
it to work to help their organizations run more efficiently, but at the same time, executives are right to be somewhat cautious, recognizing that these are
still early days and they have to learn through experimentation if this is truly transformative technology.']
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, Grey, DuckDuckGo, privacy invasion, inappropriate data handling, organic
search results, ad systems affecting organic results, Danny Sullivan, X, The Age, Google, WIRED, ad revenue
</item 3>

<item 4>
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
</item 4>

<item 5>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, Megan Grey, semantic keyword matching, The Age, Google
</item 5>

<item 6>
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
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
the list of blue links in search results that are not sponsored. “Ad keyword matching is a long-standing and well-known process that is designed to connect
people to relevant ads. A separate process, which has nothing to do with ads, is used to match organic results to a query,” he said. “It’s no secret that Google
Search looks beyond the specific words in a query to better understand their meaning ... If you make a spelling mistake, or search for a term that’s not on a
page but where the page has a close synonym, or if you aren’t even sure exactly how to search for something, our meaning matching systems help.” It is true that
some search queries result in more ads than others, and what ads you get can vary wildly depending on your request, which is the result of the keyword matching
Google uses to sell ads. But Google denies pushing users towards the most monetisable subjects.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> ads
- European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’ -MENTIONS-> ads
- Twitch’s money guy talks about the revenue split controversy and its monetization long game -MENTIONS-> ads
- Expedia -SAID-> ads
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search

Entities: ads, Google Search, Is Google Search better than the rest? And is that fair?, organic results, keyword matching, The Age, Google
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=3f4b6538f45aebfb ===
