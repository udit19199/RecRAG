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

=== ITEM id=134edb6d251fac14 ===
Gold sentences and facts:
Sentence 0: “Users’ natural behavior for search is to follow the path of least resistance, i.e.
  Fact 0.0: Users' natural behavior for search is to follow the path of least resistance.
Sentence 1: “We make Google be the default search engine,” he said, “because we’ve always thought it was the best.
  Fact 1.0: A speaker said 'We make Google be the default search engine.'
  Fact 1.1: The speaker said the reason was 'because we've always thought it was the best.'

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, ISA, John Schmidtlein, Google / Apple partnership,
Google Search, Apple, Google, Safari, The Verge
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

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> US v. Google

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, ISA, privacy pop-ups, search engines, Google Search, DuckDuckGo,
Apple, Google, trial, The Verge
</item 2>

<item 3>
Graph query row:
techCrunchArticle: Deft is building e-commerce search to help you find the right product quickly
techCrunchPublisher: TechCrunch
techCrunchPassages: ['Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)\nE-commerce search
startup Deft is releasing a multimodal search function for users to get better results. The company, which opened its service to beta users in May, has rolled
out the new search experience so users don’t have to describe the item in words.\n\nThe company is trying to solve the problem of finding the right item without
spending hours looking for a product. Often users end up with subpar results while searching for products because their search query description won’t match the
details of listed items. The startup is relying on different techniques, including building its own parser and knowledge graph combined with large language
models (LLMs) and multimodal search.\n\nDeft was founded in 2019 by Alex Gunnarson and Zach Hudson. Gunnarson started working on the idea after he couldn’t find
a specific type of couch online even after putting in different parameters and descriptions. It took him hours to find a perfect result for his search, so
Gunnarson decided to build a solution with Hudson.', 'Deft is building e-commerce search to help you find the right product quickly (TechCrunch,
2023-11-03T13:00:14+00:00)\nUsers can go to Deft and search for things like “wooden lamps with three legs under $400” or upload an image, and modify their
search to “lamps under $400 that look like this.” While it’s hard to know you can do this if someone doesn’t tell you that it works, users can also paste the
image directly into the search bar. The company is working on highlighting the image search feature better.\n\nThe company said that it doesn’t just look at the
listing and description for search results. It also looks at materials, reviews, photos and more. This way, when someone searches for a “pet-friendly” couch,
Deft can return results with stain and scratch-resistant material.\n\n“You have companies like Google and Amazon that continue to push the most relevant
listings down so they can make more advertising dollars.” Zach Hudson\n\nDeft is currently focusing on furniture and home décor search but aims to expand to
different categories along the way. The company has built its own scraping tech to add products to its search results.', 'Deft is building e-commerce search to
help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)\nThe company has built its own scraping tech to add products to its search
results. It shows results from Target, Pottery Barn and other places on the web. The company hasn’t added searches from Amazon yet because it feels there are a
lot of “junk” listings on the site and it will take some work to clean up that data.\n\nHudson told TechCrunch that there are a few plaguing problems with the
traditional search experience that prevent users from finding a suitable item.\n\n“You have companies like Google and Amazon that continue to push the most
relevant listings down so they can make more advertising dollars. Some easy searches are now hard because marketers have learned how to game these search
engines. Plus, now you have AI content that is putting the burden on users to separate fact from fiction,” he said.\n\nThe company raised $1.8 million in a seed
round from Hustle Fund, Frontier VC, Vitalize, Long Ecommerce Ventures, 43 Ventures and angel investors.', 'Deft is building e-commerce search to help you find
the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)\nHowever, the startup wants subscriptions to be its primary revenue driver. Deft is currently
testing two paid tiers: a $10 a month plan, which will offer better search customization to filter out some brands, and a $20 per month plan, which will enable
personal shopping service, that will continue to look for suitable products even after you stop searching manually.\n\nUp next, Deft aims to release a browser
extension that helps users search for products by the end of this year or early next year. In the next three years, the company wants to build and release
search APIs that other e-commerce sites could use.\n\nChallenges of building a search startup\n\nGiven that it’s building a search solution for e-commerce, Deft
already faces challenges from Amazon and Google. In September, Amazon updated its visual search feature that lets users snap a photo of a product and search for
similar items on the platform. Google released its multi-search feature last year that lets users fuse images and text.', 'Deft is building e-commerce search to
help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)\nGoogle released its multi-search feature last year that lets users fuse images
and text. Separately, Deft will also face competition from other startups like Objective, a Matrix Partners-backed startup building an API platform for
multimodal search.\n\nCaroline Casson, a partner at Vitalize Venture Capital and an investor in Deft, believes the startup has an edge because of its knowledge
graph for products.\n\n“Building a robust e-commerce search engine is incredibly complex on the backend. The extensive knowledge graph that Deft has built
combined with AI search capabilities makes for a high-quality user experience. We were excited about early iterations of the product when we invested and we
continue to be impressed with each product enhancement,” Casson told TechCrunch over email.\n\nAdii Pienaar, another investor, and co-founder of WooCommerce,
which was acquired by Automattic, told TechCrunch over a call that he sees Deft as solving user experience problems.\n\n“Over the last couple of years,
consumers have been trying out new ways to interact with and purchase products.', 'Deft is building e-commerce search to help you find the right product quickly
(TechCrunch, 2023-11-03T13:00:14+00:00)\nI see Deft as creating a new kind of UX for purchasing,” he said.\n\nStartups like Neeva — which was founded by former
Google execs — tried to convince people to switch to paid and private search with additional features. However, earlier this year, the startup shut down its
consumer efforts and eventually got acquired by Snowflake.\n\nPienaar said that Deft has to give compelling reasons to users to switch to its
service.\n\n“Users’ natural behavior for search is to follow the path of least resistance, i.e. most people just pick up their phone and just use the default
search engine. For Deft to change that behavior, they will need to create a high-value proposition,” he said.\n\nAmir Konigsberg, an investor and founder of
e-commerce search startup Twiggle (which shut down), said his startup wanted to solve ChatGPT-type interactions in the shopping world. He said that the main
problem for startups in this space is the go-to-market strategy and fundraising.', 'Deft is building e-commerce search to help you find the right product
quickly (TechCrunch, 2023-11-03T13:00:14+00:00)\nHe said that the main problem for startups in this space is the go-to-market strategy and fundraising.\n\n“The
problem with competing with e-commerce players is not just about user experience, it’s also about logistics infrastructure and supply chain. So a startup
catering to consumer search has to invest a lot of money into making it a viable business,” Konigsberg told TechCrunch.\n\nHe also added that it will be
difficult for a startup like Deft to build user loyalty while competing against Google and Amazon.\n\nBoth Pienaar and Konigsberg said that Deft will have to
involve retailers in the mix either as partners or as customers of their search product. Either way, it’s going to be interesting to see if Deft can build a
user base on its own or sell its search feature to e-commerce retailers.']
vergeArticle: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’
vergePublisher: The Verge
vergePassages: ['Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\nEddy Cue, in a dark suit,
peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses with only binders, but now
the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted down at the
screen.\n\n“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit
Mehta, presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody
got back down to business.\n\nCue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.', 'Apple defends
Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\nCue is one of the highest-profile witnesses in the case
so far, in part because the deal between Google and Apple — which makes Google the default search engine on all Apple devices and pays Apple billions of dollars
a year — is central to the US Department of Justice’s case against Google.\n\nCue had two messages: Apple believes in protecting its users’ privacy, and it also
believes in Google. Whether those two statements can be simultaneously true became the question of the day.\n\nApple is in court because of something called the
Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s products. The ISA has been in place since 2002, but Cue
was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In testimony today, the Justice Department grilled Cue about the
specifics of the deal.\n\nWhen the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users
it directed toward the search engine.', 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)\nDiscussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai
wanted to keep the deal as it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount
since.\n\n“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”\n\nMeagan Bellshaw, a Justice Department lawyer,
asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure. Cue said he’d never really considered that an
option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also argued that the deal was about more than
economics and that Apple never seriously considered switching to another provider or building its own search product. “Certainly there wasn’t a valid
alternative to Google at the time,” Cue said. He said there still isn’t one.', 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’
(The Verge, 2023-09-26T17:01:22+00:00)\nHe said there still isn’t one.\n\nThat question — whether Apple picked Google because it’s the most lucrative choice or
the best product — was a key part of Cue’s testimony and, in fact, a key part of the DOJ’s entire case against Google. The Justice Department is focused on the
deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every
platform.\n\nBellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when
you first boot up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up
that tells you when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple
offers its users a choice about lots of things, big and small, and that search could be one of them.', 'Apple defends Google Search deal in court: ‘There wasn’t
a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\n“We try to get people up and running as fast as possible.”\n\nCue acknowledged that the ISA didn’t
allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We try to get people up and running as
fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of would just be a bad user experience,
he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default search engine,” he said, “because we’ve
always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of contention in this trial — DuckDuckGo’s
CEO, who testified last week, claimed it takes “too many steps” to switch.)\n\nAs for the privacy pop-ups? This is where Bellshaw began to press on how exactly
Apple decided Google had the best product.', 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)\nThis is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user
privacy is important, to which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy
policies. Cue readily agreed. “We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was
that Google had to allow people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or
anyone else to track users.\n\nBellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to
everything Apple believes is important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue
to review some of Apple’s financial filings.', 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)\nBellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of
Apple’s profits, she asked? Cue said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing
that an agreement like this could work as well as it does.\n\nLater, after a closed session in the courtroom and a break for lunch, Google lawyer John
Schmidtlein led Cue through a history of the Google / Apple partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and
search bar was a user interface innovation, and the seamless Google integration was part of what made it work. In early promotional materials for Safari,
Schmidtlein pointed out, the Google integration was nearly always mentioned.', 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’
(The Verge, 2023-09-26T17:01:22+00:00)\n“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in
the URL field, or you could type in another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to
search.”\n\nSchmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated
seamlessly with Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching
search engines is so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.\n\nAt least, that’s all
the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries about revealing
confidential numbers and corporate secrets.', 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)\nBut the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it
just the one writing the biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought
about it. Google said Apple would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.']
sharedMentionedEntities: ['Google']
</item 3>

<item 4>
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

Graph facts:
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> Apple's own search product
- Cue -SAID-> search engines
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: US Department of Justice, Information Services Agreement, Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Sundar
Pichai, Google Search, Apple, Google, privacy, The Verge
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

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> US v. Google

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, Yahoo, Schmidtlein, search engines, Google Search, Bing,
Apple, Google, trial, Safari, The Verge
</item 5>

<item 6>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, ISA, Google Search deal, Google’s privacy policies,
Apple’s financial filings, Google Search, Apple, DOJ, Google, user privacy, Safari, The Verge
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> US v. Google

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, App Tracking Transparency, Cue’s testimony, location
tracking, Mozilla, Google Search, Apple, Maps, data collection, Samsung, Google, search, Justice Department
</item 7>

<item 8>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
I see Deft as creating a new kind of UX for purchasing,” he said.

Startups like Neeva — which was founded by former Google execs — tried to convince people to switch to paid and private search with additional features.
However, earlier this year, the startup shut down its consumer efforts and eventually got acquired by Snowflake.

Pienaar said that Deft has to give compelling reasons to users to switch to its service.

“Users’ natural behavior for search is to follow the path of least resistance, i.e. most people just pick up their phone and just use the default search engine.
For Deft to change that behavior, they will need to create a high-value proposition,” he said.

Amir Konigsberg, an investor and founder of e-commerce search startup Twiggle (which shut down), said his startup wanted to solve ChatGPT-type interactions in
the shopping world. He said that the main problem for startups in this space is the go-to-market strategy and fundraising.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, TechCrunch, Google, e-commerce search, Deft, Deft is building e-commerce search to help you find the right product quickly, Neeva, Snowflake,
Pienaar, Amir Konigsberg, Twiggle, UX for purchasing, go-to-market strategy, fundraising, ChatGPT-type interactions
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

Graph facts:
- Meagan Bellshaw -WORKS_FOR-> U.S. Justice Department
- Cue -SAID-> search engines
- Cue -PARTICIPATED_IN-> Cue’s testimony
- 5 things we learned so far about the Google antitrust case -MENTIONS-> Meagan Bellshaw
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue

Entities: Cue, Meagan Bellshaw, Apple's own search product, Google-Apple revenue-share deal, Apple defends Google Search deal in court: ‘There wasn’t a valid
alternative’, Google Search, Apple, Pichai, Google, Justice Department, The Verge
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, search engine market, Google Search, Apple, Google, Justice
Department, The Verge
</item 10>
=== END ITEM id=134edb6d251fac14 ===

=== ITEM id=98e77efff212cc56 ===
Gold sentences and facts:
Sentence 0: Tyreek Hill had two touchdowns among his 157 receiving yards to help the Miami Dolphins rout the Washington Commanders (4-9).
  Fact 0.0: Tyreek Hill had two touchdowns in this game.
  Fact 0.1: Tyreek Hill had 157 receiving yards in this game.
  Fact 0.2: Tyreek Hill helped the Miami Dolphins rout the Washington Commanders.
  Fact 0.3: The Washington Commanders were 4-9 at this point.
Sentence 1: • Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games.
  Fact 1.0: Tyreek Hill made six catches for 163 yards and a touchdown against Carolina.
  Fact 1.1: Tyreek Hill had 812 yards through the first six games.
Sentence 2: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 2.0: Miami has five remaining opponents.
  Fact 2.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.

Retrieved items:
<item 1>
Graph query row:
article: Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver
publisher: Sporting News
passage: Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
In the final weeks of the fantasy football regular season, rolling out a strong receiver corps comprised of wideouts set up for high usage days is every fantasy
owner's goal. Having those WRs in plus matchups makes it even more likely you'll end the week in the win column. Our Week 13 fantasy WR PPR rankings can help
pinpoint all the studs and sleepers at fantasy's deepest position.

There aren't as many studs or sleepers this week, as six byes and some inopportune injuries thin out our rankings. Nonetheless, we still have to fill out the
same number of starting spots in our lineups, so those in 12-team leagues might have to dig deep to find decent WR3s and flexes. That's where we come in.

Now that we know the backdrop to this pivotal week, let's break down our Week 13 WR rankings.

MORE: Watch NFL RedZone live with Sling (50% off first month)

Which WRs are on bye in Week 13?
connections: [{'entity': 'Davante Adams', 'relationship': 'MENTIONS'}, {'entity': 'Jakobi Meyers', 'relationship': 'MENTIONS'}, {'entity': 'Kalif Raymond',
'relationship': 'MENTIONS'}, {'entity': 'Raiders', 'relationship': 'MENTIONS'}, {'entity': 'NFL', 'relationship': 'MENTIONS'}, {'entity': 'Christian Kirk',
'relationship': 'MENTIONS'}, {'entity': 'Chris Olave', 'relationship': 'MENTIONS'}, {'entity': 'Jaguars', 'relationship': 'MENTIONS'}, {'entity': 'Vikings',
'relationship': 'MENTIONS'}, {'entity': 'Joe Burrow', 'relationship': 'MENTIONS'}, {'entity': 'Saints', 'relationship': 'MENTIONS'}, {'entity': 'Week 13',
'relationship': 'MENTIONS'}, {'entity': 'Week 12', 'relationship': 'MENTIONS'}, {'entity': 'Jayden Reed', 'relationship': 'MENTIONS'}, {'entity': 'Puka Nacua',
'relationship': 'MENTIONS'}, {'entity': 'Bills', 'relationship': 'MENTIONS'}, {'entity': 'Tank Dell', 'relationship': 'MENTIONS'}, {'entity': 'Detroit',
'relationship': 'MENTIONS'}, {'entity': 'New England', 'relationship': 'MENTIONS'}, {'entity': 'Titans', 'relationship': 'MENTIONS'}, {'entity': 'Bengals',
'relationship': 'MENTIONS'}, {'entity': 'Jordan Addison', 'relationship': 'MENTIONS'}, {'entity': 'Packers', 'relationship': 'MENTIONS'}, {'entity': 'McLaurin',
'relationship': 'MENTIONS'}, {'entity': 'Cooper Kupp', 'relationship': 'MENTIONS'}, {'entity': 'Jalin Hyatt', 'relationship': 'MENTIONS'}, {'entity': 'Standard
rankings', 'relationship': 'MENTIONS'}, {'entity': 'NFC Championship rematch', 'relationship': 'MENTIONS'}, {'entity': 'Michael Pittman Jr.', 'relationship':
'MENTIONS'}, {'entity': 'Lynn Bowden Jr.', 'relationship': 'MENTIONS'}, {'entity': 'A.T.', 'relationship': 'MENTIONS'}, {'entity': 'Bowden Jr.', 'relationship':
'MENTIONS'}, {'entity': 'A.T. Perry', 'relationship': 'MENTIONS'}, {'entity': 'Keith Kirkwood', 'relationship': 'MENTIONS'}, {'entity': 'Mingo', 'relationship':
'MENTIONS'}, {'entity': 'Tillman', 'relationship': 'MENTIONS'}, {'entity': 'Jalen Ramsey', 'relationship': 'MENTIONS'}, {'entity': 'Sauce Gardner',
'relationship': 'MENTIONS'}, {'entity': 'PPR leagues', 'relationship': 'MENTIONS'}, {'entity': 'Josh Palmer', 'relationship': 'MENTIONS'}, {'entity': 'PPR
scoring formats', 'relationship': 'MENTIONS'}, {'entity': 'DeVante Parker', 'relationship': 'MENTIONS'}, {'entity': 'Curtis Samuel', 'relationship':
'MENTIONS'}, {'entity': 'Rondale Moore', 'relationship': 'MENTIONS'}, {'entity': 'Texans', 'relationship': 'MENTIONS'}, {'entity': 'Skyy Moore', 'relationship':
'MENTIONS'}, {'entity': 'Dontayvion Wicks', 'relationship': 'MENTIONS'}, {'entity': 'Trenton Irwin', 'relationship': 'MENTIONS'}, {'entity': 'Nick
Westbrook-Ikhine', 'relationship': 'MENTIONS'}, {'entity': 'Cedrick Wilson', 'relationship': 'MENTIONS'}, {'entity': 'Marquez Valdes-Scantling', 'relationship':
'MENTIONS'}, {'entity': 'Chris Moore', 'relationship': 'MENTIONS'}, {'entity': 'WR PPR rankings', 'relationship': 'MENTIONS'}, {'entity': 'fantasy projections',
'relationship': 'MENTIONS'}, {'entity': 'rankings', 'relationship': 'MENTIONS'}, {'entity': 'Cardinals', 'relationship': 'MENTIONS'}, {'entity': 'C.J. Stroud',
'relationship': 'MENTIONS'}, {'entity': 'Dolphins', 'relationship': 'MENTIONS'}, {'entity': 'Eagles', 'relationship': 'MENTIONS'}, {'entity': 'wide receivers',
'relationship': 'MENTIONS'}, {'entity': 'Demario Douglas', 'relationship': 'MENTIONS'}, {'entity': 'P.J. Walker', 'relationship': 'MENTIONS'}, {'entity':
'Dallas', 'relationship': 'MENTIONS'}, {'entity': 'Deebo Samuel', 'relationship': 'MENTIONS'}, {'entity': 'DeVonta Smith', 'relationship': 'MENTIONS'},
{'entity': 'Justin Herbert', 'relationship': 'MENTIONS'}, {'entity': 'Nico Collins', 'relationship': 'MENTIONS'}, {'entity': 'Mike Evans', 'relationship':
'MENTIONS'}, {'entity': 'fantasy football', 'relationship': 'MENTIONS'}, {'entity': 'PPR rankings', 'relationship': 'MENTIONS'}, {'entity': 'Marquise Brown',
'relationship': 'MENTIONS'}, {'entity': 'Wide receiver', 'relationship': 'MENTIONS'}, {'entity': 'PPR', 'relationship': 'MENTIONS'}, {'entity': 'Rashid
Shaheed', 'relationship': 'MENTIONS'}, {'entity': 'DK Metcalf', 'relationship': 'MENTIONS'}, {'entity': 'DJ Moore', 'relationship': 'MENTIONS'}, {'entity':
'Jaylen Waddle', 'relationship': 'MENTIONS'}, {'entity': 'DeAndre Hopkins', 'relationship': 'MENTIONS'}, {'entity': 'Amari Cooper', 'relationship': 'MENTIONS'},
{'entity': 'Calvin Ridley', 'relationship': 'MENTIONS'}, {'entity': 'Zay Flowers', 'relationship': 'MENTIONS'}, {'entity': 'Noah Brown', 'relationship':
'MENTIONS'}, {'entity': 'Courtland Sutton', 'relationship': 'MENTIONS'}, {'entity': 'Romeo Doubs', 'relationship': 'MENTIONS'}, {'entity': 'Chris Godwin',
'relationship': 'MENTIONS'}, {'entity': 'Ravens', 'relationship': 'MENTIONS'}, {'entity': 'Diontae Johnson', 'relationship': 'MENTIONS'}, {'entity': 'Elijah
Moore', 'relationship': 'MENTIONS'}, {'entity': 'Josh Downs', 'relationship': 'MENTIONS'}, {'entity': 'George Pickens', 'relationship': 'MENTIONS'}, {'entity':
'Drake London', 'relationship': 'MENTIONS'}, {'entity': 'Robert Woods', 'relationship': 'MENTIONS'}, {'entity': 'Gabe Davis', 'relationship': 'MENTIONS'},
{'entity': 'Tyler Boyd', 'relationship': 'MENTIONS'}, {'entity': 'Odell Beckham Jr.', 'relationship': 'MENTIONS'}, {'entity': 'Jonathan Mingo', 'relationship':
'MENTIONS'}, {'entity': 'K.J. Osborn', 'relationship': 'MENTIONS'}, {'entity': 'JuJu Smith-Schuster', 'relationship': 'MENTIONS'}, {'entity': 'DJ Chark',
'relationship': 'MENTIONS'}, {'entity': 'Khalil Shakir', 'relationship': 'MENTIONS'}, {'entity': 'Alec Pierce', 'relationship': 'MENTIONS'}, {'entity': 'Treylon
Burks', 'relationship': 'MENTIONS'}, {'entity': 'Cedric Tillman', 'relationship': 'MENTIONS'}, {'entity': 'Miami', 'relationship': 'MENTIONS'}, {'entity':
'Kadarius Toney', 'relationship': 'MENTIONS'}, {'entity': 'Jets', 'relationship': 'MENTIONS'}, {'entity': 'Garrett Wilson', 'relationship': 'MENTIONS'},
{'entity': 'Chiefs', 'relationship': 'MENTIONS'}, {'entity': 'Lions', 'relationship': 'MENTIONS'}, {'entity': 'Amon-Ra St. Brown', 'relationship': 'MENTIONS'},
{'entity': 'Justin Jefferson', 'relationship': 'MENTIONS'}, {'entity': "Ja'Marr Chase", 'relationship': 'MENTIONS'}, {'entity': 'Jerry Jeudy', 'relationship':
'MENTIONS'}, {'entity': 'Washington', 'relationship': 'MENTIONS'}, {'entity': 'Browns', 'relationship': 'MENTIONS'}, {'entity': 'Rams', 'relationship':
'MENTIONS'}, {'entity': 'Tee Higgins', 'relationship': 'MENTIONS'}, {'entity': 'Geno Smith', 'relationship': 'MENTIONS'}, {'entity': 'Seahawks', 'relationship':
'MENTIONS'}, {'entity': 'Stefon Diggs', 'relationship': 'MENTIONS'}, {'entity': 'Jahan Dotson', 'relationship': 'MENTIONS'}, {'entity': 'Commanders',
'relationship': 'MENTIONS'}, {'entity': 'Terry McLaurin', 'relationship': 'MENTIONS'}, {'entity': 'Jaxon Smith-Njigba', 'relationship': 'MENTIONS'}, {'entity':
'Tyler Lockett', 'relationship': 'MENTIONS'}, {'entity': 'Chargers', 'relationship': 'MENTIONS'}, {'entity': 'A.J. Brown', 'relationship': 'MENTIONS'},
{'entity': 'CeeDee Lamb', 'relationship': 'MENTIONS'}, {'entity': 'Brandin Cooks', 'relationship': 'MENTIONS'}, {'entity': 'Cowboys', 'relationship':
'MENTIONS'}, {'entity': 'Giants', 'relationship': 'MENTIONS'}, {'entity': 'Cleveland', 'relationship': 'MENTIONS'}, {'entity': 'Tutu Atwell', 'relationship':
'MENTIONS'}, {'entity': 'Christian Watson', 'relationship': 'MENTIONS'}, {'entity': 'Buccaneers', 'relationship': 'MENTIONS'}, {'entity': 'Patriots',
'relationship': 'MENTIONS'}, {'entity': 'Broncos', 'relationship': 'MENTIONS'}, {'entity': 'Panthers', 'relationship': 'MENTIONS'}, {'entity': 'Colts',
'relationship': 'MENTIONS'}, {'entity': 'Adam Thielen', 'relationship': 'MENTIONS'}, {'entity': 'NFL RedZone', 'relationship': 'MENTIONS'}, {'entity': 'Sling',
'relationship': 'MENTIONS'}, {'entity': 'Bears', 'relationship': 'MENTIONS'}, {'entity': 'Draft Sharks', 'relationship': 'MENTIONS'}, {'entity': 'Free-Agent
Finder', 'relationship': 'MENTIONS'}, {'entity': 'Trade Navigator', 'relationship': 'MENTIONS'}, {'entity': '49ers', 'relationship': 'MENTIONS'}, {'entity':
'Brandon Aiyuk', 'relationship': 'MENTIONS'}, {'entity': 'Keenan Allen', 'relationship': 'MENTIONS'}, {'entity': 'Stroud', 'relationship': 'MENTIONS'},
{'entity': 'Rashee Rice', 'relationship': 'MENTIONS'}, {'entity': 'Falcons', 'relationship': 'MENTIONS'}, {'entity': 'Seattle', 'relationship': 'MENTIONS'},
{'entity': 'Jake Browning', 'relationship': 'MENTIONS'}, {'entity': 'DFS', 'relationship': 'MENTIONS'}, {'entity': 'Steelers', 'relationship': 'MENTIONS'}]
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, The New York Times, University of Tennessee, Super Bowl, DeAndre Hopkins, Lions, Rams, Tyreek Hill, N.F.L. Executives on the Biggest Questions
Facing Super Bowl Contenders, Don Hutson, Wes Welker, Elroy Hirsch, Santana Moss, Raymond Berry, Jerry Rice
</item 2>

<item 3>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Taylor Swift, Travis Kelce, Patrick Mahomes, Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in
NFL’s Century-Old Game Plan Towards Global Recognition?, Global recognition, EssentiallySports, Colin Kaepernick, Union of Concerned Scientists, racism,
domestic violence by players, Deflategate, Bountygate, CTE, pro football
</item 3>

<item 4>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, New York Jets, Miami Dolphins, Tyreek Hill, Yardbarker, Calvin Johnson, Quincy Williams, Quinnen, Pro Football Focus, AFC, MVPs for all 16 AFC
teams at the midpoint
</item 4>

<item 5>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Philadelphia Eagles
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Philadelphia Eagles
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Philadelphia Eagles
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Philadelphia Eagles
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Philadelphia Eagles

Entities: Philadelphia Eagles, The New York Times, Travis Kelce, Mahomes, Jalen Hurts, Super Bowl, Chiefs, N.F.L., Chris Jones, explosive play rates, explosive
passing, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders
</item 5>

<item 6>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Patrick Mahomes, Super Bowl, N.F.L., TruMedia, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders, Myles
Garrett, Kansas City Chiefs
</item 6>

<item 7>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Saints
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Saints
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Saints
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Saints

Entities: Saints, Week 13, Week 12, Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver, Dallas, Justin Herbert, Mike Evans,
fantasy football, Wide receiver, Miami, Lions, Sporting News, Amon-Ra St. Brown, Washington, Seahawks
</item 7>

<item 8>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.

Graph facts:
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Puka Nacua
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Puka Nacua
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> Puka Nacua
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Puka Nacua
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Puka Nacua

Entities: Puka Nacua, CBSSports.com, Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more, DeSean Jackson, Deebo Samuel, Tyreek
Hill, Tyler Lockett
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 9>

<item 10>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
1 seed in the conference.

Compared to the 49ers, Cowboys and Eagles, Goff and the Lions may have the easiest schedule to finish the season.

But bettors should also pay attention to C.J. Stroud. He leads the league in passing yards, is one of four quarterbacks eho have started the whole season with a
passer rating over 100, and his touchdown to interception ratio is the best in the NFL (4 to 1)

As for the value of his play, Stroud has taken one of the worst teams in the NFL (11 total wins the last three seasons) and turned it into a playoff contender
and potential division champion.

Offensive Player of the Year

Current favorite: Tyreek Hill

Longshot: CeeDee Lamb

There's a good reason why Tyreek Hill is such a heavy favorite to win OPOY. The Dolphins wide receiver is on track to become the first player with 2,000-plus
receiving yards in a single season.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Goff
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Goff
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Goff
- Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments -MENTIONS-> Goff
- Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Goff

Entities: Goff, NFL, C.J. Stroud, Dolphins, Eagles, OPOY, DPOY, NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more, Lions,
Sporting News, Tyreek Hill, CeeDee Lamb, Cowboys, 49ers, MVP
</item 10>
=== END ITEM id=98e77efff212cc56 ===
