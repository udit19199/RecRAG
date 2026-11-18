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

=== ITEM id=8939f55c89c680e7 ===
Gold sentences and facts:
Sentence 0: “Certainly there wasn’t a valid alternative to Google at the time,” Cue said.
  Fact 0.0: Cue said there was certainly no valid alternative to Google at the time.
Sentence 1: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 1.0: Google receives other benefits from Android.
  Fact 1.1: Google could solve this problem today if it wanted to.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content, their readers and ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, ISA, Google Search deal, Google’s privacy policies,
Apple’s financial filings, Google Search, Apple, DOJ, Google, user privacy, Safari, The Verge
</item 4>

<item 5>
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
</item 5>

<item 6>
Graph query row:
article: Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’
date: 2023-09-26T17:01:22+00:00
publisher: The Verge
named_entities: ['Eddy Cue', 'Amit Mehta', 'US v. Google', 'Mac', 'US Department of Justice', 'Information Services Agreement', 'Cue', 'Meagan Bellshaw',
"Apple's own search product", 'Google-Apple revenue-share deal', 'Bellshaw', 'App Tracking Transparency', 'Cue’s testimony', 'location tracking', 'ISA',
'privacy pop-ups', 'Google Search deal', 'Google’s privacy policies', 'Apple’s financial filings', 'John Schmidtlein', 'Google / Apple partnership', 'Yahoo',
'Schmidtlein', 'search engines', 'search engine market', 'Sundar Pichai', 'Mozilla', 'Google Search', 'DuckDuckGo', 'antitrust', 'Bing', 'Apple', 'Washington,
DC', 'Maps', 'data collection', 'Pichai', 'DOJ', 'Samsung', 'Google', 'trial', 'search', 'Justice Department', 'privacy', 'user privacy', 'iPhones', 'iPhone',
'Safari']
passages: [{'page': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)', 'passage': 'Apple
defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\nEddy Cue, in a dark suit, peered down at the
monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses with only binders, but now the tech was up
and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted down at the screen.\n\n“The resolution on
this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta, presiding over the case,
leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got back down to
business.\n\nCue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.'}, {'page': 'Apple defends Google
Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)', 'passage': 'Apple defends Google Search deal in court: ‘There
wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\nCue is one of the highest-profile witnesses in the case so far, in part because the deal
between Google and Apple — which makes Google the default search engine on all Apple devices and pays Apple billions of dollars a year — is central to the US
Department of Justice’s case against Google.\n\nCue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether
those two statements can be simultaneously true became the question of the day.\n\nApple is in court because of something called the Information Services
Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s products. The ISA has been in place since 2002, but Cue was responsible for
negotiating its current iteration with Google CEO Sundar Pichai in 2016. In testimony today, the Justice Department grilled Cue about the specifics of the
deal.\n\nWhen the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward
the search engine.'}, {'page': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)',
'passage': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\nDiscussion of specific numbers
was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as it was. They eventually
compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.\n\n“I always felt like it was in Google’s best
interest, and our best interest, to get a deal done.”\n\nMeagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if
the two sides couldn’t agree on a revenue-share figure. Cue said he’d never really considered that an option: “I always felt like it was in Google’s best
interest, and our best interest, to get a deal done.” Cue also argued that the deal was about more than economics and that Apple never seriously considered
switching to another provider or building its own search product. “Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there
still isn’t one.'}, {'page': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)', 'passage':
'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\nHe said there still isn’t one.\n\nThat
question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key part
of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many
others — to ensure it is the default search engine on practically every platform.\n\nBellshaw asked Cue a number of questions about the iPhone setup process.
Those three screenshots showed the Appearance screen that shows up when you first boot up your iPhone so you can pick font sizes; the location-tracking prompt
that appears when you open Maps; and the App Tracking Transparency pop-up that tells you when an app wants to collect your data. Cue objected to all these
things being considered part of setup, but Bellshaw’s point was that Apple offers its users a choice about lots of things, big and small, and that search could
be one of them.'}, {'page': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)', 'passage':
'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\n“We try to get people up and running as
fast as possible.”\n\nCue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to
do that anyway. “We try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines
they’ve never heard of would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make
Google be the default search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is
a persistent point of contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)\n\nAs for the privacy
pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.'}, {'page': 'Apple defends Google Search deal in court:
‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)', 'passage': 'Apple defends Google Search deal in court: ‘There wasn’t a valid
alternative’ (The Verge, 2023-09-26T17:01:22+00:00)\nThis is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked
Cue if Apple believes user privacy is important, to which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed
against Google’s privacy policies. Cue readily agreed. “We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of
the ISA with Google was that Google had to allow people to search without logging in and that Apple has done things in Safari and around its platforms to make
it harder for Google or anyone else to track users.\n\nBellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a
privacy menace anathema to everything Apple believes is important to its users, but Apple gives it a central place in its platform because Google pays it so
handsomely. Bellshaw asked Cue to review some of Apple’s financial filings.'}, {'page': 'Apple defends Google Search deal in court: ‘There wasn’t a valid
alternative’ (The Verge, 2023-09-26T17:01:22+00:00)', 'passage': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)\nBellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of
Apple’s profits, she asked? Cue said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing
that an agreement like this could work as well as it does.\n\nLater, after a closed session in the courtroom and a break for lunch, Google lawyer John
Schmidtlein led Cue through a history of the Google / Apple partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and
search bar was a user interface innovation, and the seamless Google integration was part of what made it work. In early promotional materials for Safari,
Schmidtlein pointed out, the Google integration was nearly always mentioned.'}, {'page': 'Apple defends Google Search deal in court: ‘There wasn’t a valid
alternative’ (The Verge, 2023-09-26T17:01:22+00:00)', 'passage': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)\n“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field,
or you could type in another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to
search.”\n\nSchmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated
seamlessly with Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching
search engines is so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.\n\nAt least, that’s all
the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries about revealing
confidential numbers and corporate secrets.'}, {'page': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)', 'passage': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge,
2023-09-26T17:01:22+00:00)\nBut the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it
just the one writing the biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought
about it. Google said Apple would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.'}]
google_claims: [{'claim': 'How ego and fear fuelled the rise of artificial intelligence', 'relationship': 'MENTIONS'}, {'claim': 'ChatGPT: Everything you need
to know about the AI-powered chatbot', 'relationship': 'MENTIONS'}, {'claim': 'test', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'There’s something going on
with AI startups in France', 'relationship': 'MENTIONS'}, {'claim': 'One year later, ChatGPT is still alive and kicking', 'relationship': 'MENTIONS'}, {'claim':
'How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement', 'relationship': 'MENTIONS'}, {'claim': 'regulation', 'relationship': 'SAID'}, {'claim':
'Gemini', 'relationship': 'SAID'}, {'claim': 'Hassabis', 'relationship': 'WORKS_FOR'}, {'claim': 'Victoria Krakovna', 'relationship': 'WORKS_FOR'}, {'claim':
'auction', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'London', 'relationship': 'LOCATED_IN'}, {'claim': 'AlphaGo', 'relationship': 'MADE'}, {'claim':
'Gemini', 'relationship': 'MADE'}, {'claim': 'Google', 'relationship': 'OWNS'}, {'claim': 'Google', 'relationship': 'PART_OF'}, {'claim': 'DeepMind ethics
board', 'relationship': 'PART_OF'}, {'claim': 'Demis Hassabis', 'relationship': 'FOUNDED'}, {'claim': 'Mustafa Suleyman', 'relationship': 'FOUNDED'}, {'claim':
'Hassabis', 'relationship': 'FOUNDED'}, {'claim': 'Suleyman', 'relationship': 'FOUNDED'}, {'claim': 'A comprehensive list of 2023 tech layoffs', 'relationship':
'MENTIONS'}, {'claim': '6 VCs explain how startups can capture and defend marketshare in the AI era', 'relationship': 'MENTIONS'}, {'claim': 'Apple',
'relationship': 'MENTIONS'}, {'claim': 'AI Principles', 'relationship': 'MENTIONS'}, {'claim': 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and
TikTok tests an ad-free tier', 'relationship': 'MENTIONS'}, {'claim': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’',
'relationship': 'MENTIONS'}, {'claim': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'relationship':
'MENTIONS'}, {'claim': 'Why Mozilla is betting on a decentralized social networking future', 'relationship': 'MENTIONS'}, {'claim': 'Wired opinion article',
'relationship': 'MENTIONS'}, {'claim': 'Is Google Search better than the rest? And is that fair?', 'relationship': 'MENTIONS'}, {'claim': 'News publisher files
class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'claim': 'What to know about Fortnite maker
Epic Games’ antitrust battle with Google, starting today', 'relationship': 'MENTIONS'}, {'claim': 'Founders, are events useful?', 'relationship': 'MENTIONS'},
{'claim': 'Passwords? We don’t need no steenkin’ passwords', 'relationship': 'MENTIONS'}, {'claim': 'Govee’s colorful LED light strips are on sale just in time
for Halloween', 'relationship': 'MENTIONS'}, {'claim': 'From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray', 'relationship':
'MENTIONS'}, {'claim': 'The best Black Friday deals you can get for under $25', 'relationship': 'MENTIONS'}, {'claim': 'Everton players and fans will be fired
up by points penalty, asserts Sean Dyche', 'relationship': 'MENTIONS'}, {'claim': 'Discogs’ vibrant vinyl community is shattering', 'relationship': 'MENTIONS'},
{'claim': 'Biden’s AI EO hailed as broad, but not deep without legislation to match', 'relationship': 'MENTIONS'}, {'claim': 'Chelsea secure last-eight spot
after Raheem Sterling sinks Blackburn', 'relationship': 'MENTIONS'}, {'claim': 'Why We Desperately Need a New (and Better) Google', 'relationship': 'MENTIONS'},
{'claim': 'The people who ruined the internet', 'relationship': 'MENTIONS'}, {'claim': 'People are turning to Snap Map for firsthand perspectives from Gaza',
'relationship': 'MENTIONS'}, {'claim': 'Early impressions of Google’s Gemini aren’t great', 'relationship': 'MENTIONS'}, {'claim': 'The restaurant nearest
Google', 'relationship': 'MENTIONS'}, {'claim': 'Meta seeks legislation that would require parents to approve teens’ app downloads', 'relationship':
'MENTIONS'}, {'claim': 'Tim Sweeney on Epic’s victory royale over Google', 'relationship': 'MENTIONS'}, {'claim': 'England thrash Scotland but cannot avoid Team
GB Olympic heartbreak', 'relationship': 'MENTIONS'}, {'claim': '‘Mass breach of privacy’: TikTok under fire for tracking users online', 'relationship':
'MENTIONS'}, {'claim': 'If your startup doesn’t seem impossible at first, it’s not hard enough', 'relationship': 'MENTIONS'}, {'claim': 'The best Prime Day
smart home deals', 'relationship': 'MENTIONS'}, {'claim': 'England choose survival over oblivion and a chance of summer trip to France', 'relationship':
'MENTIONS'}, {'claim': 'The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more', 'relationship': 'MENTIONS'}, {'claim': "Sex, Signal
messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court", 'relationship': 'MENTIONS'}, {'claim': 'The best VPN services for
2023', 'relationship': 'MENTIONS'}, {'claim': 'Keep Labs’ smart storage redefines medication management', 'relationship': 'MENTIONS'}, {'claim': 'The best
October Prime Day gaming deals', 'relationship': 'MENTIONS'}, {'claim': '5 things we learned so far about the Google antitrust case', 'relationship':
'MENTIONS'}, {'claim': 'Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew', 'relationship': 'MENTIONS'}, {'claim': 'Deepfake
election risks trigger EU call for more generative AI safeguards', 'relationship': 'MENTIONS'}, {'claim': 'The 30 best Amazon Prime Day deals for October 2023',
'relationship': 'MENTIONS'}, {'claim': 'Meta and IBM form an AI Alliance, but to what end?', 'relationship': 'MENTIONS'}, {'claim': 'NMPA: Generative AI is ‘the
greatest risk to the human creative class that has ever existed’', 'relationship': 'MENTIONS'}, {'claim': 'The Epic v. Google trial may come down to simple v.
complicated', 'relationship': 'MENTIONS'}, {'claim': 'Deft is building e-commerce search to help you find the right product quickly', 'relationship':
'MENTIONS'}, {'claim': "Sweeping White House AI executive order takes aim at the technology's toughest challenges", 'relationship': 'MENTIONS'}, {'claim':
'These Are Our Favorite Walmart Cyber Monday Deals', 'relationship': 'MENTIONS'}, {'claim': 'Ten Hag demands both positivity and pragmatism from erratic Onana',
'relationship': 'MENTIONS'}, {'claim': 'Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs', 'relationship': 'MENTIONS'}, {'claim':
'Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more', 'relationship': 'MENTIONS'}, {'claim': 'European roundup: Bellingham helps Real
Madrid go top while Bayern draw', 'relationship': 'MENTIONS'}, {'claim': 'Epic v. Google, explained', 'relationship': 'MENTIONS'}, {'claim': 'The Ray-Ban Meta
smart glasses actually make the future look cool', 'relationship': 'MENTIONS'}, {'claim': '8 predictions for AI in 2024', 'relationship': 'MENTIONS'}, {'claim':
'Here are the best deals on headphones and wireless earbuds we could find', 'relationship': 'MENTIONS'}, {'claim': 'Ten Hag’s Manchester United stand up and die
at club stuck in the past tense', 'relationship': 'MENTIONS'}, {'claim': 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned
from Dennis Kooker’s speech about AI', 'relationship': 'MENTIONS'}, {'claim': 'Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI
experiment', 'relationship': 'MENTIONS'}, {'claim': 'Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment.', 'relationship':
'MENTIONS'}, {'claim': 'Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US
Copyright Office.', 'relationship': 'MENTIONS'}, {'claim': 'Temu was the most-downloaded iPhone app in the U.S. in 2023', 'relationship': 'MENTIONS'}, {'claim':
'Know the price-matching policies for Best Buy, Target, Walmart, and others', 'relationship': 'MENTIONS'}, {'claim': 'The best Black Friday headphone deals from
Sony, Apple, Bose, Beats and more', 'relationship': 'MENTIONS'}, {'claim': 'Israel’s startup ecosystem: down but not out', 'relationship': 'MENTIONS'},
{'claim': 'Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference',
'relationship': 'MENTIONS'}, {'claim': 'French soccer players back African mobile video network StarNews Mobile in $3M round', 'relationship': 'MENTIONS'},
{'claim': 'EU commissioner sidesteps MEPs’ questions about CSAM microtargeting', 'relationship': 'MENTIONS'}, {'claim': 'The new, ‘efficient’ Spotify has a very
different approach to podcasting', 'relationship': 'MENTIONS'}, {'claim': 'ASX set to rise ahead of inflation report after big tech lifts Wall Street',
'relationship': 'MENTIONS'}, {'claim': 'Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne', 'relationship': 'MENTIONS'},
{'claim': 'The UK just laid out new rules for the internet — it only gets harder from here', 'relationship': 'MENTIONS'}, {'claim': 'Tesla ‘digs its own grave
with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle', 'relationship': 'MENTIONS'}, {'claim': '78 Absolute Best October Prime Day Deals
(2023)', 'relationship': 'MENTIONS'}, {'claim': 'EU says incoming rules for general purpose AIs can evolve over time', 'relationship': 'MENTIONS'}, {'claim':
"India to be one of world's fastest growing economies over next decade", 'relationship': 'MENTIONS'}, {'claim': 'Cruise hits the brakes on driverless, UAW makes
progress and more EV backpedaling', 'relationship': 'MENTIONS'}, {'claim': 'Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter',
'relationship': 'MENTIONS'}, {'claim': 'The 26 best Amazon Prime Day deals you can still shop for today', 'relationship': 'MENTIONS'}, {'claim': 'Some of our
favorite devices are on sale for Black Friday.', 'relationship': 'MENTIONS'}, {'claim': 'The best anti-Prime Big Deal Day deals', 'relationship': 'MENTIONS'},
{'claim': 'Pebble, the Twitter alternative previously known as T2, is shutting down', 'relationship': 'MENTIONS'}, {'claim': 'The best Prime Day deals on
headphones for October 2023', 'relationship': 'MENTIONS'}, {'claim': 'Palestine’s growing tech industry has been literally blown apart by Israel’s war on
Hamas', 'relationship': 'MENTIONS'}, {'claim': 'The best apps and tools for managing your money online', 'relationship': 'MENTIONS'}, {'claim': 'Is Google
Search better than the rest? And is that fair?', 'relationship': 'SAID'}, {'claim': 'App store competition', 'relationship': 'SAID'}, {'claim': 'Chat logs and
documents provided to the court', 'relationship': 'SAID'}, {'claim': 'workforce reductions', 'relationship': 'SAID'}, {'claim': 'open source AI',
'relationship': 'SAID'}, {'claim': 'Apple', 'relationship': 'SAID'}, {'claim': 'misinformation', 'relationship': 'SAID'}, {'claim': 'Bard', 'relationship':
'SAID'}, {'claim': 'Gemini Pro', 'relationship': 'SAID'}, {'claim': 'Thai food near me', 'relationship': 'SAID'}, {'claim': 'local search', 'relationship':
'SAID'}, {'claim': 'bias', 'relationship': 'SAID'}, {'claim': 'Activision competing store', 'relationship': 'SAID'}, {'claim': 'Epic', 'relationship': 'SAID'},
{'claim': 'ethical reviews', 'relationship': 'SAID'}, {'claim': 'security', 'relationship': 'SAID'}, {'claim': 'safety', 'relationship': 'SAID'}, {'claim':
'Meta’s open source AI approach', 'relationship': 'SAID'}, {'claim': 'AI training', 'relationship': 'SAID'}, {'claim': 'Android sideloading flow',
'relationship': 'SAID'}, {'claim': 'AI-generated content', 'relationship': 'SAID'}, {'claim': 'sideloaded apps equal gaping security holes', 'relationship':
'SAID'}, {'claim': 'fair use', 'relationship': 'SAID'}, {'claim': 'YouTube', 'relationship': 'SAID'}, {'claim': 'Artificial Intelligence and Copyright',
'relationship': 'SAID'}, {'claim': 'ingestion of copyrighted material for training generative AI platforms', 'relationship': 'SAID'}, {'claim': 'US copyright
law', 'relationship': 'SAID'}, {'claim': 'copyright harvesting', 'relationship': 'SAID'}, {'claim': 'privacy', 'relationship': 'SAID'}, {'claim': 'generative
AI', 'relationship': 'SAID'}, {'claim': 'Cue', 'relationship': 'SAID'}, {'claim': 'Rosenberg', 'relationship': 'SAID'}, {'claim': 'Barry Diller',
'relationship': 'SAID'}, {'claim': 'Mitchell Baker', 'relationship': 'SAID'}, {'claim': 'Suleyman', 'relationship': 'SAID'}, {'claim': 'Megan Grey',
'relationship': 'SAID'}, {'claim': 'Helena World Chronicle', 'relationship': 'SAID'}, {'claim': 'Epic Games', 'relationship': 'SAID'}, {'claim': 'federal
government', 'relationship': 'SAID'}, {'claim': 'Forrester', 'relationship': 'SAID'}, {'claim': 'Sullivan', 'relationship': 'SAID'}, {'claim': 'SEOs',
'relationship': 'SAID'}, {'claim': 'Ray', 'relationship': 'SAID'}, {'claim': 'Cutts', 'relationship': 'SAID'}, {'claim': 'Lee', 'relationship': 'SAID'},
{'claim': 'Sacha Haworth', 'relationship': 'SAID'}, {'claim': 'Tim Sweeney', 'relationship': 'SAID'}, {'claim': 'Don Harrison', 'relationship': 'SAID'},
{'claim': 'Samantha Hoffman', 'relationship': 'SAID'}, {'claim': 'Wilkins', 'relationship': 'SAID'}, {'claim': 'U.S. government', 'relationship': 'SAID'},
{'claim': 'Pichai', 'relationship': 'SAID'}, {'claim': 'DOJ', 'relationship': 'SAID'}, {'claim': 'Gary Bornstein', 'relationship': 'SAID'}, {'claim':
'Bornstein', 'relationship': 'SAID'}, {'claim': 'Pomerantz', 'relationship': 'SAID'}, {'claim': 'Zach Hudson', 'relationship': 'SAID'}, {'claim': 'Justice
Department', 'relationship': 'SAID'}, {'claim': 'Gary Wang', 'relationship': 'WORKS_FOR'}, {'claim': 'John Schmidtlein', 'relationship': 'WORKS_FOR'}, {'claim':
'Jamie Rosenberg', 'relationship': 'WORKS_FOR'}, {'claim': 'Sundar Pichai', 'relationship': 'WORKS_FOR'}, {'claim': 'Hinton', 'relationship': 'WORKS_FOR'},
{'claim': 'Alan Eustace', 'relationship': 'WORKS_FOR'}, {'claim': 'Kent Walker', 'relationship': 'WORKS_FOR'}, {'claim': 'Matt Cutts', 'relationship':
'WORKS_FOR'}, {'claim': 'Danny Sullivan', 'relationship': 'WORKS_FOR'}, {'claim': 'Sullivan', 'relationship': 'WORKS_FOR'}, {'claim': 'Cutts', 'relationship':
'WORKS_FOR'}, {'claim': 'Prabhakar Raghavan', 'relationship': 'WORKS_FOR'}, {'claim': 'David Drummond', 'relationship': 'WORKS_FOR'}, {'claim': 'Glenn
Pomerantz', 'relationship': 'WORKS_FOR'}, {'claim': 'Ilya Sutskever', 'relationship': 'WORKS_FOR'}, {'claim': 'Lewis', 'relationship': 'WORKS_FOR'}, {'claim':
'Bradley Horowitz', 'relationship': 'WORKS_FOR'}, {'claim': 'Dalia Awad', 'relationship': 'WORKS_FOR'}, {'claim': 'Google-Apple revenue-share deal',
'relationship': 'PARTICIPATED_IN'}, {'claim': 'Google antitrust lawsuit with Match Group', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Google trial with Epic
Games', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'US v Google trial', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'quality tests', 'relationship':
'PARTICIPATED_IN'}, {'claim': 'Epic Games lawsuit against Google over app store antitrust issues', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Epic Games’
antitrust battle with Google', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'deal between Google and Samsung', 'relationship': 'PARTICIPATED_IN'}, {'claim':
'Google developer agreements', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'three-year agreement', 'relationship': 'PARTICIPATED_IN'}, {'claim': "Epic's
antitrust lawsuit", 'relationship': 'PARTICIPATED_IN'}, {'claim': "Google's antitrust lawsuit with the Department of Justice", 'relationship':
'PARTICIPATED_IN'}, {'claim': 'antitrust battle with Google', 'relationship': 'PARTICIPATED_IN'}, {'claim': '2023 I/O event', 'relationship':
'PARTICIPATED_IN'}, {'claim': 'Activision-Google deal', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Epic’s victory royale over Google', 'relationship':
'PARTICIPATED_IN'}, {'claim': 'Epic v. Google', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'U.S. government’s antitrust trial against Google',
'relationship': 'PARTICIPATED_IN'}, {'claim': 'lucrative search agreement', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Talks to secure licenses for artists’
vocals and musical melodies', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'trial', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Epic sued Google',
'relationship': 'PARTICIPATED_IN'}, {'claim': 'Epic v. Google trial', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'tentative settlement over app store
antitrust claims', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'settlement', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'US v. Google antitrust trial',
'relationship': 'PARTICIPATED_IN'}, {'claim': "Epic's trial", 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Epic v. Google lawsuit', 'relationship':
'PARTICIPATED_IN'}, {'claim': 'Google’s submission to the United States Copyright Office', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Google’s recent filing
with the US Copyright Office', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Google filing with the US Copyright Office', 'relationship': 'PARTICIPATED_IN'},
{'claim': 'Google USCO submission', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'Google’s USCO filing', 'relationship': 'PARTICIPATED_IN'}, {'claim': 'US v
Google trial', 'relationship': 'ACCUSED_OF'}, {'claim': 'anticompetitive behavior', 'relationship': 'ACCUSED_OF'}, {'claim': 'U.S. antitrust law',
'relationship': 'ACCUSED_OF'}, {'claim': 'Sherman Act', 'relationship': 'ACCUSED_OF'}, {'claim': 'class action antitrust suit', 'relationship': 'ACCUSED_OF'},
{'claim': 'misappropriation of Publishers’ content', 'relationship': 'ACCUSED_OF'}, {'claim': 'Epic Games lawsuit against Google over app store antitrust
issues', 'relationship': 'ACCUSED_OF'}, {'claim': 'class action antitrust suit against Google', 'relationship': 'ACCUSED_OF'}, {'claim': 'digital ad
technologies', 'relationship': 'ACCUSED_OF'}, {'claim': 'anti-competitive practices', 'relationship': 'ACCUSED_OF'}, {'claim': 'search monopoly',
'relationship': 'ACCUSED_OF'}, {'claim': 'antitrust', 'relationship': 'ACCUSED_OF'}, {'claim': 'monopoly', 'relationship': 'ACCUSED_OF'}, {'claim':
'anti-competition law', 'relationship': 'ACCUSED_OF'}, {'claim': 'systematically destroying chats', 'relationship': 'ACCUSED_OF'}, {'claim': 'anticompetitive
conduct', 'relationship': 'ACCUSED_OF'}, {'claim': 'antitrust law', 'relationship': 'ACCUSED_OF'}, {'claim': 'unlawful monopoly', 'relationship': 'ACCUSED_OF'},
{'claim': 'making potential evidence disappear', 'relationship': 'ACCUSED_OF'}, {'claim': 'illegal tying', 'relationship': 'ACCUSED_OF'}, {'claim': 'Android app
distribution', 'relationship': 'ACCUSED_OF'}, {'claim': 'Android in-app payment processing', 'relationship': 'ACCUSED_OF'}, {'claim': 'deleting potentially
incriminating messages', 'relationship': 'ACCUSED_OF'}, {'claim': 'Epic', 'relationship': 'ACCUSED_OF'}, {'claim': 'US', 'relationship': 'LOCATED_IN'},
{'claim': 'Israel', 'relationship': 'LOCATED_IN'}, {'claim': 'Gmail', 'relationship': 'MADE'}, {'claim': 'Pixel 8', 'relationship': 'MADE'}, {'claim': 'Chrome',
'relationship': 'MADE'}, {'claim': 'Play', 'relationship': 'MADE'}, {'claim': 'Google travel listings', 'relationship': 'MADE'}, {'claim': 'Google Search',
'relationship': 'MADE'}, {'claim': 'User Choice Billing', 'relationship': 'MADE'}, {'claim': 'Google first-party billing system', 'relationship': 'MADE'},
{'claim': 'Google Play', 'relationship': 'MADE'}, {'claim': 'Search', 'relationship': 'MADE'}, {'claim': 'Google Pixel Buds Pro', 'relationship': 'MADE'},
{'claim': 'Google Pixel Watch', 'relationship': 'MADE'}, {'claim': 'Bard', 'relationship': 'MADE'}, {'claim': 'Gemini Pro', 'relationship': 'MADE'}, {'claim':
'Google Maps', 'relationship': 'MADE'}, {'claim': 'Google One VPN', 'relationship': 'MADE'}, {'claim': 'Google-developed AI tools', 'relationship': 'MADE'},
{'claim': 'enterprise-focused AI services', 'relationship': 'MADE'}, {'claim': 'multi-search', 'relationship': 'MADE'}, {'claim': 'multi-search feature',
'relationship': 'MADE'}, {'claim': 'Nest Cam', 'relationship': 'MADE'}, {'claim': 'Gemini Ultra', 'relationship': 'MADE'}, {'claim': 'Google Pixel Buds
A-Series', 'relationship': 'MADE'}, {'claim': 'Google Cloud Platform', 'relationship': 'MADE'}, {'claim': 'Google reCaptcha', 'relationship': 'MADE'}, {'claim':
'Google Nest Thermostat', 'relationship': 'MADE'}, {'claim': 'Google Nest Hub (second-gen)', 'relationship': 'MADE'}, {'claim': 'Gmail', 'relationship':
'OWNS'}, {'claim': 'Google Search', 'relationship': 'OWNS'}, {'claim': 'Google Play', 'relationship': 'OWNS'}, {'claim': 'Google Play Store', 'relationship':
'OWNS'}, {'claim': 'DeepMind', 'relationship': 'OWNS'}, {'claim': 'Bard', 'relationship': 'OWNS'}, {'claim': 'Riot', 'relationship': 'OWNS'}, {'claim': 'Google
Maps', 'relationship': 'OWNS'}, {'claim': 'Gemini', 'relationship': 'OWNS'}, {'claim': 'YouTube', 'relationship': 'OWNS'}, {'claim': 'Alphabet', 'relationship':
'OWNS'}, {'claim': 'Yahoo', 'relationship': 'COMPETES_WITH'}, {'claim': 'DuckDuckGo', 'relationship': 'COMPETES_WITH'}, {'claim': 'ChatGPT', 'relationship':
'COMPETES_WITH'}, {'claim': 'Bing', 'relationship': 'COMPETES_WITH'}, {'claim': 'Apple', 'relationship': 'COMPETES_WITH'}, {'claim': 'Reddit', 'relationship':
'COMPETES_WITH'}, {'claim': 'large language models', 'relationship': 'COMPETES_WITH'}, {'claim': 'Brave', 'relationship': 'COMPETES_WITH'}, {'claim': 'OpenAI',
'relationship': 'COMPETES_WITH'}, {'claim': 'TikTok', 'relationship': 'COMPETES_WITH'}, {'claim': 'Epic', 'relationship': 'COMPETES_WITH'}, {'claim': 'IBM',
'relationship': 'COMPETES_WITH'}, {'claim': 'Deft', 'relationship': 'COMPETES_WITH'}, {'claim': 'TicWatch Pro 5', 'relationship': 'COMPETES_WITH'}, {'claim':
'Alphabet', 'relationship': 'PART_OF'}, {'claim': 'Partnership on AI', 'relationship': 'PART_OF'}, {'claim': 'Alphabet Inc.', 'relationship': 'PART_OF'},
{'claim': 'Gradient Ventures', 'relationship': 'PART_OF'}, {'claim': 'DeepMind', 'relationship': 'PART_OF'}, {'claim': 'Google Users & Products',
'relationship': 'PART_OF'}, {'claim': 'Area 120', 'relationship': 'PART_OF'}, {'claim': 'web spam team', 'relationship': 'PART_OF'}, {'claim': 'Larry Page',
'relationship': 'FOUNDED'}, {'claim': 'Sergey Brin', 'relationship': 'FOUNDED'}]
</item 6>

<item 7>
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

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Satya Nadella, DuckDuckGo, Bing, The Information, Microsoft, Apple, TechCrunch, Tim Cook, Pichai, Google antitrust case, 5 things we learned
so far about the Google antitrust case, Google, iOS, iPhone
</item 7>

<item 8>
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

Graph facts:
- Grey -WORKS_FOR-> DuckDuckGo
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> DuckDuckGo
- Is Google Search better than the rest? And is that fair? -MENTIONS-> DuckDuckGo
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> antitrust

Entities: third-party developers, DuckDuckGo, antitrust, Bing, Microsoft, Apple, monopoly, TechCrunch, data, Google antitrust case, Brave, Microsoft trial,
anticompetitive conduct, APIs, computer manufacturers
</item 8>

<item 9>
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
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Mac
- The best video games of the year so far -MENTIONS-> Mac
- Amit Mehta -PARTICIPATED_IN-> US v Google trial
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- The best VPN services for 2023 -MENTIONS-> Mac

Entities: Eddy Cue, Amit Mehta, US v. Google, Mac, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Google Search, antitrust,
Apple, Washington, DC, Google, iPhones, The Verge
</item 10>
=== END ITEM id=8939f55c89c680e7 ===
