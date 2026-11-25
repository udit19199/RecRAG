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

=== ITEM id=87bd7b4cd5b2cf53 ===
Gold sentences and facts:
Sentence 0: While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s.
  Fact 0.0: Altman is not trained as an AI engineer.
  Fact 0.1: Altman is now 38 years old.
  Fact 0.2: Altman has been seen as a Silicon Valley wunderkind since his early 20s.
Sentence 1: Altman has been telling investors that he’s planning to launch a new venture, according to The Information.
  Fact 1.0: Altman has been telling investors that he is planning to launch a new venture.
  Fact 1.1: The information about Altman's planned new venture comes from The Information.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The theory is based on the board's language.
  Fact 2.1: The theory is based on the way these giant tech companies work.
  Fact 2.2: The theory is the prevailing theory floating around right now.

Retrieved items:
<item 1>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 1>

<item 2>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI CEO Sam Altman had previously pointed to industry-wide GPU shortages for the service’s spotty performance. Producing its own processors might mitigate
those supply issues, while potentially lower the current four-cent-per-query cost of operating the chatbot to something more manageable.

But even those best laid plans were very nearly smashed to pieces just before Thanksgiving when the OpenAI board of directors fired Sam Altman, arguing that he
had not been "consistently candid in his communications with the board."

That firing didn't take. Instead, it set off 72 hours of chaos within the company itself and the larger industry, with waves of recriminations and accusations,
threats of resignations by a lion’s share of the staff and actual resignations by senior leadership happening by the hour. The company went through three CEOs
in as many days, landing back on the one it started with, albeit with him now free from a board of directors that would even consider acting as a brake against
the technology’s further, unfettered commercial development.
</item 2>

<item 3>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Confirmation Sam Altman will not return as OpenAI’s CEO

A number of investors and OpenAI employees tried to bring back Altman after his sudden firing by the company’s board, but following a weekend of negotiations,
it was confirmed that Altman would not return to OpenAI and new leadership would take hold. What this means for ChatGPT’s future, and for the OpenAI Dev Day
announcements, remains to be seen.

Sam Altman ousted as OpenAI’s CEO

Sam Altman has been fired from OpenAI. He will leave the company’s board and step down as CEO, with OpenAI’s chief technology officer Mira Murati stepping in as
interim CEO. In a blog post from OpenAI, the company writes that the board “no longer has confidence in [Altman’s] ability to continue leading OpenAI.”

In a statement on X, Altman said working at OpenAI “was transformative” for him and “hopefully the world.”
</item 3>

<item 4>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.
</item 4>

<item 5>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
In a dramatic turn of events late Friday, ex-Y Combinator president Sam Altman was fired as CEO of AI startup OpenAI, the company behind viral AI hits like
ChatGPT, GPT-4 and DALL-E 3, by OpenAI’s board of directors. Then, the company’s longtime president and co-founder, Greg Brockman, resigned — as did three
senior OpenAI researchers. And the fallout continues.

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

It’s a fast-moving situation that we’re still trying to get to the bottom of. No doubt more will become clear as time goes on. To make it easier to follow all
that’s happened in the meantime, though, we’ve put together a timeline; we’ll do our best to keep it current.
</item 5>

<item 6>
OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch, 2023-12-18T20:12:25+00:00)
Of course, the result of said drama was the sidelining of two of the more critical voices and the appointment of some money-minded guys (Bret Taylor and Larry
Summers), who are sharp but not AI experts by a long shot.

If a panel of experts makes a recommendation, and the CEO makes decisions based on that information, will this friendly board really feel empowered to
contradict them and hit the brakes? And if they do, will we hear about it? Transparency is not really addressed outside a promise that OpenAI will solicit
audits from independent third parties.

Say a model is developed that warrants a “critical” risk category. OpenAI hasn’t been shy about tooting its horn about this kind of thing in the past — talking
about how wildly powerful their models are, to the point where they decline to release them, is great advertising. But do we have any kind of guarantee this
will happen, if the risks are so real and OpenAI is so concerned about them? Maybe it’s a bad idea. But either way it isn’t really mentioned.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 7>

<item 8>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“This is another indication that AI is progressing faster than even many experts anticipated,” Victoria Krakovna, who would soon join DeepMind as an “AI safety”
researcher, wrote in a blog post. DeepMind’s founders were increasingly worried about what Google would do with their inventions. In 2017, they tried to break
away from the company. Google responded by increasing the salaries and stock award packages of the DeepMind founders and their staff. They stayed put. The
ethics board never had a second meeting. The Breakup

Convinced that Page’s optimistic view of AI was dead wrong, and angry at his loss of DeepMind, Musk built his own lab. OpenAI was founded in late 2015, just a
few months after he met with Altman at the Rosewood hotel in Silicon Valley. Sam Altman Credit: Bloomberg Businessweek Musk pumped money into the lab, and his
former PayPal buddies – Hoffman and Thiel – came along for the ride.
</item 8>

<item 9>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
On Equity this week, the crew talked about the SBF trial; deals from VR firms Rainforest, At One Ventures, Section 32 and Greylock, where venture funding has
declined; and how Fearless Fund, a firm founded to invest in women of color, is being barred from awarding grants to Black women founders.

Meanwhile, Found featured Esther Rodriguez-Villegas from Acurable, a medical device company that makes patient-friendly wearable devices to diagnose and manage
respiratory conditions at home. As a career-long academic, Rodriguez-Villegas talks about how she never intended to be a founder until she learned about how the
currently available medical devices make it extremely difficult to detect and treat diseases like sleep apnea and epilepsy.

And over on Chain Reaction, Jacquelyn did a crossover episode with Alex about the SBF trial.
</item 9>

<item 10>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Altman and others pushed back, Musk quit and said he would focus on his own AI work at Tesla. In February 2018, he announced his departure to OpenAI’s
staff on the top floor of the startup’s offices in a converted truck factory, three people who attended the meeting said. When he said that OpenAI needed to
move faster, one researcher retorted at the meeting that Musk was being reckless. Musk called the researcher a “jackass” and stormed out, taking his deep
pockets with him. OpenAI suddenly needed new financing in a hurry. Altman flew to Sun Valley for a conference and ran into Satya Nadella, Microsoft’s CEO. A
tie-up seemed natural. Altman knew Microsoft’s chief technology officer, Kevin Scott. Microsoft had bought LinkedIn from Hoffman, an OpenAI board member.
Nadella told Scott to get it done. The deal closed in 2019.
</item 10>

<item 11>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
one takeaway: go tell your friends how great you think they are. — Sam Altman (@sama) November 18, 2023

Investors pushing for Altman’s return

Investors — furious at the turn of events — are reportedly exerting pressure on OpenAI’s board to reinstate Altman, going so far as to recruit Microsoft.
Nadella is said to be sympathetic.

Board agrees to reverse course — in principle

The Verge reports that the board agreed in principle to resign and to allow Altman and Brockman to return. It waffled, however, missing a deadline yesterday by
which many OpenAI staffers were set to leave the company. Altman is said to be ambivalent about coming back and asking for “significant” governance changes.

—

November 17

Brockman demoted

Brockman says he got a text from Sutskever shortly after noon on Friday asking for a quick call.
</item 11>

<item 12>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He’s sat with multiple heads of state to discuss AI’s potential and perils. Just Thursday, he took part in a CEO summit at the Asia-Pacific Economic Cooperation
conference in San Francisco, where OpenAI is based.

He predicted AI will prove to be “the greatest leap forward of any of the big technological revolutions we’ve had so far.” He also acknowledged the need for
guardrails, calling attention to the existential dangers future AI could pose.

Some computer scientists have criticized that focus on far-off risks as distracting from the real-world limitations and harms of current AI products. The U.S.
Federal Trade Commission has launched an investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false
information through its chatbot.
</item 12>

<item 13>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
(I did observe him occasionally shaking his head, and sometimes quivering at points during her testimony, but didn’t have a view of his face.)

At one point, Ellison appeared to even be toying with Cohen

As we were approaching lunch, several jurors looked annoyed, and Cohen looked clueless. He asked Ellison to define what “buy on the way down” meant, as though
it were a term of art. (It means what you think it means, to purchase an asset that’s losing value.) This seemed to puzzle her. At other times, Cohen seemed to
forget what she had testified to, bringing up things she hadn’t said. I don’t know if this was an attempt to trap her in a lie or just poor preparation, but
much like FTX employee Adam Yedidia before her, Ellison was fastidious about making sure a question was clear and her answer was precise.

At one point, Ellison appeared to even be toying with Cohen. She’d testified on direct about Luna, a cryptocurrency token.
</item 13>

<item 14>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
There was a minimum of word salad today. I don’t know if he was more relaxed, or he’d just been more rehearsed, but I will certainly be watching to see if he
suddenly becomes much less coherent when Sassoon gets him for the cross.

Here is the story of FTX, from his point of view.

Bankman-Fried, who informed us he’s “somewhat introverted, naturally,” gave us a rather prolonged tour of his pre-Alameda Research life, which I will skip. In
2017, during a crypto bull run, he started his cryptocurrency trading firm. He knew “basically nothing” about cryptocurrency at the time, he explained, but he
wanted to do arbitrage on it anyway.

Alameda Research was named for Alameda County in California, which was where its first office was set up. As for its name, here’s what Bankman-Fried said on the
stand:

Effectively, we wanted to be under the radar at that point in time.
</item 14>

<item 15>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Any significant snafu in that department would reflect on her, as well as Altman.

5. Perhaps a difference of AI ethics or philosophy?

Altman is a proud techno-optimist, and often speaks fondly of the possibilities of AGI, or artificial general intelligence, a theoretical software system that
achieves human-like intellect and versatility.

The board’s statement pointedly includes that “OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence
benefits all humanity” and that new leadership was necessary. It’s possible that Sam’s zeal for AGI, even absent a secret project or agreement, led to a major
rift between him and the board.

It’s been obvious to all that Altman took the company in a much more corporate direction from its origins, changing its legal status and aggressively pursuing
enterprise and consumer applications. That doesn’t sound a lot like the “mission” the board wants to advance.
</item 15>

<item 16>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 16>

<item 17>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
2. Do they disagree on long-term strategy?

Despite being the hottest tech company in the world right now and everyone talking about ChatGPT, OpenAI isn’t exactly a sound business. It’s shoveling money
into the furnace as fast as it can by serving, by all accounts, a fantastically expensive product at bargain-bin prices.

That’s all well and good for a year or two, but at some point that strategy changes from a growth hack to an existential liability. Could Altman and the board
have had irreconcilable differences on where that point lies?

This doesn’t seem so likely. The company has been very deliberately pursuing this very publicly, confidently and on a long-term basis. Altman and the board seem
to be in sync on this, at least for the present.

3. Do the numbers not add up?

On the other hand, could OpenAI be losing even more money than Altman admitted or projected? It seems impossible, but the costs of running this operation have
no precedent, nor really does the operation itself.
</item 17>

<item 18>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”
</item 18>

<item 19>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He said OpenAI still has a “deep bench of technical leaders” but its next executives will have to steer it through the challenges of scaling the business and
meeting the expectations of regulators and society.

Forrester analyst Rowan Curran speculated that Altman’s departure, “while sudden,” did not likely reflect deeper business problems.

“This seems to be a case of an executive transition that was about issues with the individual in question, and not with the underlying technology or business,”
Curran said.

Altman has a number of possible next steps. Even while running OpenAI, he placed large bets on several other ambitious projects.

Among them are Helion Energy, for developing fusion reactors that could produce prodigious amounts of energy from the hydrogen in seawater, and Retro
Biosciences, which aims to add 10 years to the human lifespan using biotechnology.
</item 19>

<item 20>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
It has evolved into working in humanitarian aid, delivering emergency food supplies to people in need in Afghanistan and Turkey.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Another superconductor disappointment: Tim writes that a new, supposedly room-temperature superconducting material isn’t what the scientific community hoped it
would be. With the Nature-published paper detailing the material facing retraction, the odds of researchers discovering a room-temperature superconductor are
looking even longer.

Klarna inches toward an IPO: Mary Ann and Alex write that Swedish fintech Klarna is taking steps toward an eventual IPO. The company has initiated a process for
a legal entity restructuring to set up a holding company in the U.K.
</item 20>

<item 21>
OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch, 2023-12-18T20:12:25+00:00)
But the people making those models aren’t necessarily the best ones to evaluate them and make recommendations. For that reason, OpenAI is making a
“cross-functional Safety Advisory Group” that will sit on top of the technical side, reviewing the boffins’ reports and making recommendations inclusive of a
higher vantage. Hopefully (they say) this will uncover some “unknown unknowns,” though by their nature those are fairly difficult to catch.

The process requires these recommendations to be sent simultaneously to the board and leadership, which we understand to mean CEO Sam Altman and CTO Mira
Murati, plus their lieutenants. Leadership will make the decision on whether to ship it or fridge it, but the board will be able to reverse those decisions.

This will hopefully short-circuit anything like what was rumored to have happened before the big drama, a high-risk product or process getting greenlit without
the board’s awareness or approval.
</item 21>

<item 22>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
In a post on X (formerly Twitter), Altman wrote that he “loved his time at OpenAI” and will have “more to say about what’s next later.”

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

OpenAI’s dramatic leadership change will also see Greg Brockman — who was a member of the team that co-founded OpenAI, as was Altman — step down as chairman of
the board but remain as OpenAI’s president, reporting to the company’s newly-appointed interim CEO, Mira Murati. Murati was previously OpenAI’s CTO.

OpenAI says that it’ll begin conducting a formal search for a permanent CEO immediately.
</item 22>

<item 23>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 23>

<item 24>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
The day ended with a document that had been shared between Ellison and Bankman-Fried — with his comments appearing in bubbles along the main text. Ellison wrote
she was worried about “both actual leverage and presenting on our balance sheet.” Bankman-Fried responded with a note: “Yup, and could also get worse.”
</item 24>

<item 25>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 25>

<item 26>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 26>

<item 27>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 27>

<item 28>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 28>

<item 29>
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
</item 29>

<item 30>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 30>

<item 31>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT Voice rolls out to all free users

Even if its leadership is in flux, OpenAI is still releasing updates to ChatGPT. First announced in September and granted to paid users on a rolling basis, the
text-to-speech model can create a voice from text prompts and a few seconds of speech samples. OpenAI worked with voice actors to create the five voice options,
and you can give it a shot by heading to the settings in your mobile ChatGPT apps and tapping the “headphones” icon.

Sam Altman might return, but it’s complicated

The only constant within OpenAI right now is change, and in a series of interviews, Nadella hedged on earlier reporting that Altman and Brockman were headed to
Microsoft.

“Obviously, we want Sam and Greg to have a fantastic home if they’re not going to be in OpenAI,” Nadella said in an interview with CNBC, saying that we was
“open” to them settling at Microsoft or returning to OpenAI should the board and employees support the move.
</item 31>

<item 32>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 32>

<item 33>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 33>

<item 34>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Brockman was invited to join — but it’s unclear whether he’ll take execs up on that invitation.

Board negotiations hit a snag

Bloomberg reports that Lightcap and Murati, among others, are pushing the board to reinstate Altman. But unsurprisingly, the directors are resisting. As of
midday Sunday, the board hadn’t resigned out of concern over who could replace them, and were vetting candidates. One possible new addition could be Salesforce
co-CEO Bret Taylor.

Altman out, Shear in

Altman won’t be returning as CEO, according to a report in The Information citing an internal memo sent by Sutskever. As the search for a new permanent CEO
continues, OpenAI has appointed Emmett Shear, the co-founder of video streaming site Twitch, as interim CEO — replacing Murati.
</item 34>

<item 35>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
You can tell he had no experience in management or handling such big undertakings. I think he was in way over his head,” Ayelet Noff, the CEO and founder of
public relations firm Sliced Brand, which represents a number of cryptocurrency businesses, told TechCrunch. “I think he does a disservice to the whole
community. He pushed back crypto adoption by at least a year or two. Each time one of these things happens it brings back the whole market. It’s two steps
forward and one step back.”

The Michael Lewis interview is infuriating. SBF was a scumbag, and now we need to hear about his “tragic fall” in the MSM because Sam told them he was an
“effective altruist” and people like Michael Lewis were so far up his ass they couldn’t tell truth from fiction.
</item 35>

<item 36>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Microsoft has poured $13 billion into OpenAI so far and has what amounts to a 49% stake in the company. The former’s stock price fell more than 1% in the last
30 minutes of trading, after Altman’s departure was announced; Axios reports Microsoft was notified Altman would be leaving “minutes” before the public
announcement.

OpenAI was founded as a nonprofit in 2015, but restructured in 2019 as a “capped-profit” company to raise capital — an acknowledgement of the huge costs
associated with training cutting-edge AI systems. Underlining the point, Altman said in an interview this week with the Financial Times that he “hoped”
Microsoft would increase its investment to help pay for “huge” imminent model training costs.

When contacted via email, OpenAI’s press relations declined to comment beyond the official blog post.
</item 36>

<item 37>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
WeWork goes bust: As predicted, flexible office-space firm WeWork has filed for Chapter 11 bankruptcy protection, listing over $18.6 billion of debt in a
remarkable collapse for the once high-flying startup co-founded by Adam Neumann and bankrolled by SoftBank, BlackRock and Goldman Sachs.

Slack’s loss, Bumble’s gain: Dating app Bumble announced a doozy this week: It’s replacing founder CEO Whitney Wolfe Herd with Slack CEO Lidiane Jones. Jones
only started as CEO at Slack last year, stepping in for another founder CEO, Stewart Butterfield. Ron and Sarah write that — while Bumble now has a clear line
of succession — the move leaves Slack in a bit of a pickle.

Arrival fails to deliver: Arrival set out eight years ago to make electric vehicle production “radically more efficient.”
</item 37>

<item 38>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
We are collaborating to figure out the details. Thank you so much for your patience through this. — OpenAI (@OpenAI) November 22, 2023

Altman also posted about the new deal, giving some insight into the roller coaster that we’ve all been riding since his firing was revealed Friday. In his
words, his decision to join Microsoft on Sunday “was the best path for me and the team.” Since then, the new board’s composition and Microsoft’s support appear
to have been enough to bring him back to the AI company he co-founded.

Altman and board in talks

OpenAI’s board of directors is reportedly in talks with Sam Altman, ex-Y Combinator president and an OpenAI co-founder, to return to OpenAI as CEO as soon as
this week.
</item 38>

<item 39>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 39>

<item 40>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Sam Altman has been fired from OpenAI, Inc., the 501(c)(3) nonprofit that acts as the governing body for OpenAI, the AI startup behind ChatGPT, DALL-E 3, GPT-4
and other highly capable generative AI systems. He’ll both leave the company’s board of directors and step down as CEO.

In a post on OpenAI’s official blog, the company writes that Altman’s departure follows a “deliberative review process by the board” that concluded that Altman
“wasn’t consistently candid in his communications” with other board members, “hindering its ability to exercise its responsibilities.”

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

“The board no longer has confidence in [Altman’s] ability to continue leading OpenAI,” the blog post reads.
</item 40>

<item 41>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 41>

<item 42>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 42>

<item 43>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
When contacted via email, OpenAI’s press relations declined to comment beyond the official blog post. But Microsoft PR gave this statement from Frank Shaw, its
chief communications officer: “We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of
AI to our customers.”

Nadella later published a statement:

“As you saw at Microsoft Ignite this week, we’re continuing to rapidly innovate for this era of AI, with over 100 announcements across the full tech stack from
AI systems, models and tools in Azure, to Copilot. Most importantly, we’re committed to delivering all of this to our customers while building for the future.
We have a long-term agreement with OpenAI with full access to everything we need to deliver on our innovation agenda and an exciting product roadmap; and remain
committed to our partnership, and to Mira and the team. Together, we will continue to deliver the meaningful benefits of this technology to the world.
</item 43>

<item 44>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 44>

<item 45>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.
</item 45>

<item 46>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
60 Minutes got rightly community noted here. How is it that SBF is _still_ getting super positive media coverage when the media has nothing good to say about
the founders that have actually created things and brought value into the world? https://t.co/VPV1ORVHqf — tobi lutke (@tobi) October 2, 2023

The germination of “Going Infinite” sounds as unlikely as the story of SBF himself.

In an interview broadcast yesterday on 60 Minutes, Lewis, the famed chronicler of financial adventure and misadventure — his books include “Liar’s Poker,”
“Moneyball” and “The Big Short” — said that he didn’t set out to write a book about SBF. Rather, Lewis claims he was introduced to him in early 2021 by a friend
who wanted to back FTX. Before any financial commitment, that friend asked Lewis to meet the young billionaire — both lived in Berkeley — to “evaluate his
character.”
</item 46>

<item 47>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
SBF allegedly tried to buy off Trump: The TC team’s been trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced entrepreneur
accused of orchestrating the collapse of cryptocurrency exchange FTX. But fascinating details about SBF’s political dealings are emerging from a book by Michael
Lewis, “Going Infinite,” that debuted on the first day of the trial, like SBF’s attempt to buy off Trump to get him to not run again for president.

Gmail fights back against spammers: Google this week announced a series of significant changes to how it handles email from bulk senders in an effort to cut
down on spam and other unwanted emails. The company says that, starting next year, bulk senders will need to authenticate their emails, offer an easy way to
unsubscribe and stay under a reported spam threshold.

TikTok tests an ad-free tier: TikTok is testing an ad-free subscription tier for some users.
</item 47>

<item 48>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Nadella leaves the door open to other OpenAI staffers, saying that they’ll be given the resources they need should they choose to join.

Sutskever’s mea culpa

Sutskever publishes a post on X suggesting that he regrets his decision to remove Altman and that he’ll do everything in his power to reinstate Altman as CEO.

I deeply regret my participation in the board's actions. I never intended to harm OpenAI. I love everything we've built together and I will do everything I can
to reunite the company. — Ilya Sutskever (@ilyasut) November 20, 2023

Employees threaten to resign

Nearly 500 of OpenAI’s roughly 770 employees — including, remarkably, Sutskever — publish a letter saying that they might quit unless the startup’s board
resigns and reappoints the ousted Altman. Later Monday, that number climbed to over 650.
</item 48>

<item 49>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 49>

<item 50>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a memo internally, he published the text on X.

After learning today’s news, this is the message I sent to the OpenAI team: https://t.co/NMnG16yFmm pic.twitter.com/8x39P0ejOM — Greg Brockman (@gdb) November
18, 2023

Senior OpenAI researchers resign

Three senior OpenAI researchers resign after Brockman, including the director of research Jakub Pachocki and head of preparedness Aleksander Madry.

—

November 16

Ilya Sutskever schedules call with Altman

According to a post on X (formerly Twitter) from Brockman, Ilya Sutskever, the chief scientist at OpenAI and a co-founder, texted Altman on Thursday evening
about scheduling a Friday noon call.

Sam and I are shocked and saddened by what the board did today.
</item 50>

<item 51>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.
</item 51>

<item 52>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 52>

<item 53>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 53>

<item 54>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
This is to say nothing of the regulatory risks around crypto.

As Cohen droned on about airplanes, I couldn’t stop thinking about the missing risk officer. Bringing it up, I thought, was a tremendous mistake. The
prosecution hadn’t mentioned it. Either Bankman-Fried is stupid — unlikely — or he deliberately didn’t hire a risk officer. Was he worried about what one might
find?

Sure, as Cohen put it, Bankman-Fried was a math nerd who didn’t party. That paints a picture of someone who’s pretty deliberate, particularly since he
immediately left MIT and went to work on Wall Street. If he had been a party-hardy trainwreck, I could see overlooking a risk officer in order to do another
line, or a supermodel, or something else important. Why was the defense bringing this up?

But as Cohen tried to tell me that FTX’s and Alameda’s business relationships were “reasonable under the circumstances,” the lack of risk officer kept elbowing
me in the ribs.
</item 54>

<item 55>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“I was concerned that I had unwittingly written code that contributed to a crime,” he said.

Soon, however, the clock neared 4:30 p.m., and court adjourned for the day. Yedidia will continue his testimony on Thursday, followed by Matt Huang, a former
partner at the high-powered venture capital firm Sequoia Capital, and then Gary Wang, a key Bankman-Fried lieutenant and one of the government’s star witnesses.
</item 55>

<item 56>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a Google Meet link, Brockman was told that he was being removed from the board as chairman “but was vital to the company and would retain his
role” as president, and that Altman had been fired.

Altman’s firing publicly announced

OpenAI published a post on its blog announcing the executive shake-up. The company’s management team was aware shortly after.

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

All-hands meeting

OpenAI held an all-hands meeting Friday afternoon during which Sutskever defended Altman’s ouster. He dismissed suggestions that pushing Altman out amounted to
a “hostile takeover,” and claimed that it was necessary to protect OpenAI’s mission of “making AI beneficial to humanity.”
</item 56>

<item 57>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman also co-founded Worldcoin, a biometric and cryptocurrency project that’s been scanning people’s eyeballs with the goal of creating a vast digital
identity and financial network.

___

Associated Press business writers Haleluya Hadero in New York, Kelvin Chan in London and Michael Liedtke and David Hamilton in San Francisco contributed to this
report.
</item 57>

<item 58>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 58>

<item 59>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
as an important early step in its plans for an initial public offering, a Klarna spokesperson tells TechCrunch+.

The unicorn’s legacy isn’t over: It’s been 10 years since Cowboy Ventures’ founder Aileen Lee coined an incredibly catchy nickname for what were very rare
startups at the time: Unicorns. TechCrunch+ spoke with Lee about how she feels about the term 10 years later, now that her venture firm is also a decade old.
</item 59>

<item 60>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I was ready to hear this pursued further — but then it, too, was dropped.

I think the defense was also trying to suggest that the government had coerced Ellison’s testimony, by suggesting that she had pleaded guilty to a charge of
defrauding investors that she couldn’t have been involved in. After all, she didn’t prepare materials for them. Unfortunately, she did say that she had
conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine. This
line of questioning felt like a waste of time.

There were rather a lot of sidebars during the cross-examination, to the point that when one occurred, several jurors looked entertained. There were a few
yesterday, too, including one in which the prosecution complained that Bankman-Fried was visibly scoffing at Ellison’s answers, according to the court
transcript.
</item 60>

<item 61>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
OpenAI says that it’ll begin conducting a formal search for a permanent CEO immediately.

“OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence benefits all humanity,” the board wrote in a joint
statement. “The board remains fully committed to serving this mission. We are grateful for Sam’s many contributions to the founding and growth of OpenAI. At the
same time, we believe new leadership is necessary as we move forward. As the leader of the company’s research, product, and safety functions, Mira is
exceptionally qualified to step into the role of interim CEO. We have the utmost confidence in her ability to lead OpenAI during this transition period.”

Sam Altman is a hero of mine. He built a company from nothing to $90 Billion in value, and changed our collective world forever. I can't wait to see what he
does next. I, and billions of people, will benefit from his future work- it's going to be simply incredible.
</item 61>

<item 62>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
It has been a whirlwind four days for OpenAI, the generative AI poster child behind the smash hit ChatGPT.

Seemingly out of nowhere, the OpenAI board ousted CEO and co-founder Sam Altman and demoted president and co-founder Greg Brockman, who subsequently resigned,
paving the way for what looked like a mutiny by staff insisting the founders be reinstated post-haste. By then, Microsoft had already hired Altman and Brockman
to head up a new internal AI unit, though, as things transpired, nothing had actually been signed yet, with rumors suggesting that the ousted leaders might
actually return to OpenAI after all — in some capacity, at least.

The situation remains fluid, and any number of potential outcomes still remain on the table. But the whole debacle has shone a spotlight on the forces that
control the burgeoning AI revolution, leading many to question what happens if you go all-in on a centralized proprietary player, and what happens if things
then go belly-up?
</item 62>

<item 63>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Later Monday, that number climbed to over 650.

Altman and Brockman considering return

As reported by The Verge, Altman’s move to Microsoft isn’t a done deal — and both Altman and Brockman are still open to returning to OpenAI. That is, if the
remaining board members who initially fired him step aside.

OpenAI board considers merger

OpenAI’s board of directors approached Dario Amodei, the co-founder and CEO of rival large-language model developer Anthropic, about a potential merger of the
two companies, The Information reports. The approach was part of an effort by OpenAI to persuade Amodei to replace Altman as CEO — but Amodei quickly turned
down the CEO offer.

—

November 19

Altman to meet at OpenAI HQ

According to The Information, Altman is expected to meet at OpenAI’s San Francisco headquarters as executives at OpenAI push to have him reinstated as CEO.
</item 63>

<item 64>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 64>

<item 65>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 65>

<item 66>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 66>

<item 67>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
But later on X, formerly Twitter, Brockman posted a message he sent to OpenAI employees in which he wrote, “based on today’s news, i quit.”

In another X post on Friday night, Brockman said Altman was asked to join a video meeting at noon Friday with the company’s board members, minus Brockman,
during which OpenAI co-founder and Chief Scientist Ilya Sutskever informed Altman he was being fired.

“Sam and I are shocked and saddened by what the board did today,” Brockman wrote, adding that he was informed of his removal from the board in a separate call
with Sutskever a short time later.

OpenAI declined to answer questions on what Altman’s alleged lack of candor was about. The statement said his behavior was hindering the board’s ability to
exercise its responsibilities.

Altman posted Friday on X: “i loved my time at openai.
</item 67>

<item 68>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
That doesn’t sound a lot like the “mission” the board wants to advance. Then again, this shift didn’t happen today, and it certainly doesn’t seem like a
plausible reason for abruptly firing the CEO and a few others on a beautiful fall Friday afternoon.

6. What about IP and legal liability?

Altman told me at OpenAI’s Dev Day earlier this month that the company doesn’t want to incur any copyright problems by using (as I had asked about) datasets of
pirated books. But a lot of research I’ve been reading contradicts that, as does pretty much every AI data scientist I talk to. It’s exceedingly hard to imagine
that OpenAI built GPT-3 with the copyrighted books database (as seems to be the case) but not GPT-4 or succeeding models. (I was going to write this up next
week, so thanks OpenAI board for eating my lunch.)
</item 68>

<item 69>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 69>

<item 70>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
</item 70>

<item 71>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Or what if, and again this is purely speculation, Altman has been secretly pursuing an internal project, perhaps at significant cost, against the advice of the
board and without the necessary safety measures that probably should accompany such research? It sounds a little wild, but firing your CEO like this is also a
little wild.

Some kind of major mismatch in the financial department could be cause for dismissal, but it’s hard to imagine what Altman could have kept from the board and
CTO that would be so damning.

There’s also the possibility that Altman was making personal investments in a way that the board disagreed with. With OpenAI poised to be a kingmaker in the
field, he would certainly be in a position of power. One would think that, as an ideologically driven person already rich beyond belief and at the head of the
world’s leading AI company, Altman would have risen beyond having to do this kind of side deal, or at the very least that scrutiny on him and those close to him
would prevent them. But one can never be sure.

4.
</item 71>

<item 72>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman posted Friday on X: “i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved
working with such talented people. will have more to say about what’s next later.”

The Associated Press and OpenAI have a licensing and technology agreement allowing OpenAI access to part of the AP’s text archives.

Altman helped start OpenAI as a nonprofit research laboratory in 2015. But it was ChatGPT’s explosion into public consciousness that thrust Altman into the
spotlight as a face of generative AI — technology that can produce novel imagery, passages of text and other media. On a world tour this year, he was mobbed by
a crowd of adoring fans at an event in London.

He’s sat with multiple heads of state to discuss AI’s potential and perils.
</item 72>

<item 73>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.
</item 73>

<item 74>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
The two have since stopped speaking. Elon Musk and Sam Altman. Credit: Bloomberg, AP “There is disagreement, mistrust, egos,” Altman said. “The closer people
are to being pointed in the same direction, the more contentious the disagreements are. You see this in sects and religious orders. There are bitter fights
between the closest people.” Last month that infighting came to OpenAI’s boardroom. Rebel board members tried to force out Altman because, they believed, they
could no longer trust him to build AI that would benefit humanity. Over five chaotic days, OpenAI looked as if it were going to fall apart, until the board –
pressured by giant investors and employees who threatened to follow Altman out the door – backed down.

The drama inside OpenAI gave the world its first glimpse of the bitter feuds among those who will determine the future of AI.
</item 74>

<item 75>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
(I was going to write this up next week, so thanks OpenAI board for eating my lunch.)

If you were the board and facing the mounting accusations that your product was built on a dataset that includes thousands or millions of copyrighted works —
and your CEO had systematically downplayed the potential liability there — how would you feel? I’d feel very hurt.

But again, if copyright liability was the reason, it seems unlikely that the board would promote the CTO. Presumably, OpenAI’s Chief Science Officer Ilya
Sutskever would also have been in the know, and he’s still on the board.

7. Did CTO Mira Murati launch a coup?

Probably not — she seems cool, and anyway what CTO wants to be CEO? Mira, answer my email!

8. Was it a “personal matter”?

When someone is kicked out in a hurry, it’s not uncommon that there was some kind of unprofessional behavior in the workplace. Some CEOs get a pass on things
like having kids with direct reports, but not all.
</item 75>

<item 76>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
“He hadn’t known how to tell me,” she said.

Ellison was also, of course, in a more personal relationship with Bankman-Fried. (A juror who’d been asleep for a discussion of the FTT token woke up when she
started discussing it.) The two started sleeping together in the fall of 2018, on and off. At the time, she was a trader and Bankman-Fried was the CEO. They
didn’t date until later — twice. Their first relationship stretched from the summer of 2020 through the summer of 2021; they agreed to keep it secret. (Some
people found out, as they usually do.) The second time, from the fall of 2021 until the spring of 2022, they lived together.

That gave Ellison an unusual view of his character. “He was very ambitious,” she said.
</item 76>

<item 77>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
ChatGPT-maker Open AI said Friday it has pushed out its co-founder and CEO Sam Altman after a review found he was “not consistently candid in his
communications” with the board of directors.

“The board no longer has confidence in his ability to continue leading OpenAI,” the artificial intelligence company said in a statement.

In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial
intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.

Mira Murati, OpenAI’s chief technology officer, will take over as interim CEO effective immediately, the company said, while it searches for a permanent
replacement.

The announcement also said another OpenAI co-founder and top executive, Greg Brockman, the board’s chairman, would step down from that role but remain at the
company, where he serves as president.
</item 77>

<item 78>
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
</item 78>

<item 79>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
She’d testified on direct about Luna, a cryptocurrency token. It had a sister token, Terra, that was a paired algorithmic stablecoin. (If you don’t understand
what that means, it doesn’t matter, because they were both nonsense.) Cohen asked her about Terra/Luna and she pointed out she’d only spoken about Luna, leaving
him to fumble about how to explain the relationship between the two tokens. She kept a straight face on the stand while I chuckled from the press seats.

During the opening statement, Cohen had blamed Ellison for not taking out a hedge on some of the risks Alameda was trading. We heard more about this hedge in
cross-examination, and friends, it was stupid. Forget evaluating the trade itself (hedging being long crypto by selling Nasdaq futures). Was she supposed to
have taken more customer funds to put on the hedge? Was that the defense, that she didn’t take enough of them? Was it that she should have taken them sooner?
What the fuck?
</item 79>

<item 80>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Microsoft releases a statement

Satya Nadella, the CEO of Microsoft, a major investor in — and partner with — OpenAI, published a statement about Altman’s firing:

“As you saw at Microsoft Ignite this week, we’re continuing to rapidly innovate for this era of AI, with over 100 announcements across the full tech stack from
AI systems, models and tools in Azure, to Copilot. Most importantly, we’re committed to delivering all of this to our customers while building for the future.
We have a long-term agreement with OpenAI with full access to everything we need to deliver on our innovation agenda and an exciting product roadmap; and remain
committed to our partnership, and to Mira and the team. Together, we will continue to deliver the meaningful benefits of this technology to the world.”

Brockman quits

Brockman announced his resignation from OpenAI, citing “today’s news.”
</item 80>

<item 81>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 81>

<item 82>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 82>

<item 83>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
—

November 18

“Not … in response to malfeasance”

In an internal memo obtained by Axios sent Saturday morning, OpenAI COO Brad Lightcap said yesterday’s announcement “took [the management team] by surprise” and
that management had had “multiple conversations with the board to try to better understand the reasons and process behind their decision.” Discussions were
ongoing as of Saturday morning, per the memo.

“We can say definitively that the board’s decision was not made in response to malfeasance or anything related to our financial, business, safety, or
security/privacy practices,” Lightcap added. “This was a breakdown in communication between Sam and the board … We still share your concerns about how the
process has been handled, are working to resolve the situation, and will provide updates as we’re able.”

OpenAI’s funding in jeopardy

The planned sale of OpenAI employee shares that would value the startup at about $86 billion could be in jeopardy.
</item 83>

<item 84>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison

That “basically confirmed my suspicions that had been rising all day” that Bankman-Fried had purloined the funds, Sun said. He ran Bankman-Fried through
possible explanations, including a margin loan program, and explained that none of them could fit the actual situation — for instance, Alameda had borrowed more
from FTX than had ever been in the margin loan program. He said Bankman-Fried acknowledged their conversation with a “yup, yup.” “I was expecting a bigger
response, but it was very muted,” Sun said.

After that, Singh spilled the beans: Alameda had withdrawn assets that included the customer funds, Sun testified. He quit the next day. Like Yedidia, Sun
testified with immunity.

In the courtroom, we were treated to a Good Morning America appearance by Bankman-Fried on December 1st, 2022.
</item 84>

<item 85>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 85>

<item 86>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Couldn't be a prouder brother. — Jack Altman (@jaltma) November 17, 2023

It’s unclear what missteps Altman might’ve made helming OpenAI… at present. But they evidently relate to his relationship with OpenAI’s rather unusual board
makeup and corporate governance structure — and perhaps OpenAI’s active talks to raise substantial new capital.

As a recent piece in VentureBeat explored, OpenAI’s for-profit subsidiary, OpenAI Global, LLC, is fully controlled by OpenAI’s nonprofit. While the for-profit
subsidiary is permitted to commercialize its tech, it’s subject to the nonprofit’s mission: attaining artificial general intelligence (AGI), or AI that can — as
OpenAI defines it — “outperform humans at most economically valuable work.”

The nonprofit OpenAI’s board has the power to both determine when the company has achieved AGI and exclude this AGI from IP licenses and other commercial terms,
including with Microsoft, one of OpenAI’s biggest investors and a steadfast integrator of OpenAI’s various technologies.
</item 86>

<item 87>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
I didn’t want to call it Sam’s Crypto Trading Firm or anything like that. We — there are a lot of competitors and people who we didn’t particularly want to know
what we were building out because they would race to do it. “Research” was a sort of generic word, which filled out the company name. And that was — it was far
better than the internal name that we had at that point, which was Wireless Mouse.

I would find this much more believable if I hadn’t already watched a video of Bankman-Fried explaining on a podcast that the name made it easier to get a bank
account. That happened during the first day of Gary Wang’s testimony. Bankman-Fried was there, too. You know who else was there? The jury.

Anyway, Bankman-Fried went on a hiring spree for Alameda. He rounded up his merry gang of alleged co-conspirators. First, Wang, to program the computers. Then,
Nishad Singh, about a month after founding Alameda Research.
</item 87>

<item 88>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Sam and I are shocked and saddened by what the board did today. Let us first say thank you to all the incredible people who we have worked with at OpenAI, our
customers, our investors, and all of those who have been reaching out. We too are still trying to figure out exactly… — Greg Brockman (@gdb) November 18, 2023

Murati told of Altman’s firing

Brockman alleges that Mira Murati, OpenAI’s CTO and now interim CEO, was informed on Thursday night that Altman would be fired.
</item 88>

<item 89>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Around the same time, Musk, who had co-chaired its board with Altman, resigned from the board in a move that OpenAI said would eliminate a “potential future
conflict for Elon” due to Tesla’s work on building self-driving systems.

While OpenAI’s board has preserved its nonprofit governance structure, the startup it oversees has increasingly sought to capitalize on its technology by
tailoring its popular chatbot to business customers.

At its first developer conference last week, Altman was the main speaker showcasing a vision for a future of AI agents that could help people with a variety of
tasks. Days later, he announced the company would have to pause new subscriptions to its premium version of ChatGPT because it had exceeded capacity.

Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.
</item 89>

<item 90>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Could Altman have been working with — or against — OpenAI’s patron in secret? If Altman wanted to kill the golden goose by going independent, that might have
activated the board’s fiduciary or otherwise statutory duty. On the other hand, if he was negotiating some other deal, like an acquisition or deeper and more
exclusive integration, it could also have caused the board to bristle, either at the idea itself or at being excluded.

But if Microsoft was as shocked as the rest of us, as one report has it, it could hardly be the kind of high-stakes conspiracy some seem to be hoping for. But
one must assume that Microsoft would say that either way. Even if they’d been working with Altman on some kind of secret plan, they can truthfully say they were
surprised by his firing. (And they “remain committed to our partnership.”)

2. Do they disagree on long-term strategy?
</item 90>

<item 91>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 91>

<item 92>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 92>

<item 93>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
The company said its board consists of OpenAI’s chief scientist, Ilya Sutskever, and three non-employees: Quora CEO Adam D’Angelo, tech entrepreneur Tasha
McCauley, and Helen Toner of the Georgetown Center for Security and Emerging Technology.

OpenAI’s key business partner, Microsoft, which has invested billions of dollars into the startup and helped provide the computing power to run its AI systems,
said that the transition won’t affect its relationship.

“We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of AI to our customers,” said an
emailed Microsoft statement.

While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s. He was recruited in 2014 to take lead of
the startup incubator YCombinator.
</item 93>

<item 94>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 94>

<item 95>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
I think even swifter than we realize.” The war in the Middle East may even be the tipping point, he theorized, as it’s now “abundantly clear” that we need
something better to figure out the world than the platforms we have today.

The founder, as you may recall, notably resigned from Reddit’s board in 2020 over differences of opinion on moderation and wanting to crack down on communities
about violence and hate. It was a moment where he seemed to be reckoning with what his platform had wrought. At the time, he asked for his board seat to be
replaced by a Black board member. The company agreed, appointing its first Black board member with Y Combinator partner Michael Seibel.

Reddit also later banned several toxic communities, including r/donaldtrup following violence at the U.S. Capitol, and Ohanian says that the site has “gotten
demonstrably better,” since, as well as better as a business.
</item 95>

<item 96>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 96>

<item 97>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Thank you… — Eric Schmidt (@ericschmidt) November 17, 2023

OpenAI’s board of directors now consists of OpenAI’s chief scientist Ilya Sutskever; Quora CEO Adam D’Angelo; tech entrepreneur Tasha McCauley; and Helen Toner,
the director of strategy at Georgetown University’s Center for Security and Emerging Technology.

Altman’s removal comes unexpectedly, to say the least. The company just last week hosted its first developer conference, OpenAI DevDay, which Altman emceed.
Altman spoke at the Asia-Pacific Economic Cooperation conference and an Oakland, California event on Thursday. And according to The Verge and The New York
Times, citing multiple internal sources, OpenAI employees learned of Altman’s firing when it was announced publicly.

Altman has a long history steering OpenAI.
</item 97>

<item 98>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
But one can never be sure.

4. Could it be a major security or privacy incident?

The idea that the company has experienced a major, perhaps pervasive, security issue is bolstered by the fact that Microsoft reportedly suspended use of ChatGPT
internally a few days ago. OpenAI subsequently stopped allowing new signups. If there was a serious security problem in its biggest product and Altman
downplayed it, that would obviously create distrust with the board.

There is also the potential for misuse at scale with the enormous amount of personal data that travels through OpenAI’s APIs and services.

Working against this theory is the fact that CTO Mira Murati was just elevated to interim CEO in Altman’s place. It seems unlikely that anything
security-related would go through the CEO and not the CTO, or that the two would be at odds to the point where one could be fired like this and the other
swapped in to clean up the mess. As the board’s statement notes, Murati is in charge of product and safety, among other things.
</item 98>

<item 99>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 99>

<item 100>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That was the basis of the worst of her testimony — and the spreadsheet from hell.

Ellison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and shared it with him. The point was risk analysis around paying
back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted to use $3 billion for venture investments, so Ellison was
ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she figured there was a 30 percent chance they wouldn’t be able
to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100 percent chance they couldn’t meet the recalls, even with FTX
customer funds.

Echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length

The problem here wasn’t really the math, which seemed pretty arbitrary.
</item 100>
=== END ITEM id=87bd7b4cd5b2cf53 ===

=== ITEM id=012b9ff33c083070 ===
Gold sentences and facts:
Sentence 0: Halloween isn’t the only holiday on the horizon now that October is drawing to a close.
  Fact 0.0: Halloween is not the only holiday on the horizon.
  Fact 0.1: October is drawing to a close.
Sentence 1: If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
  Fact 1.0: The reader may be in a new relationship this holiday season.
  Fact 1.1: The reader may be faced with the opportunity to introduce their significant other to their family this holiday season.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
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
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
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
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
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
=== END ITEM id=012b9ff33c083070 ===

=== ITEM id=ecb847ecaa296e59 ===
Gold sentences and facts:
Sentence 0: While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s.
  Fact 0.0: Altman is not trained as an AI engineer.
  Fact 0.1: Altman is now 38 years old.
  Fact 0.2: Altman has been seen as a Silicon Valley wunderkind since his early 20s.
Sentence 1: Altman has been telling investors that he’s planning to launch a new venture, according to The Information.
  Fact 1.0: Altman has been telling investors that he is planning to launch a new venture.
  Fact 1.1: The information about Altman's planned new venture comes from The Information.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The theory is based on the board's language.
  Fact 2.1: The theory is based on the way these giant tech companies work.
  Fact 2.2: The theory is the prevailing theory floating around right now.

Retrieved items:

=== END ITEM id=ecb847ecaa296e59 ===

=== ITEM id=e8331ec158b98e32 ===
Gold sentences and facts:
Sentence 0: The official details have not been reported, though the report indicated it will shatter Mike Trout's record 12-year, $426.5 million deal.
  Fact 0.0: The official details have not been reported.
  Fact 0.1: The report indicated the deal will shatter Mike Trout's record 12-year, $426.5 million deal.
Sentence 1: According to ESPN's Jeff Passan, the deal is for $42 million over three years with a club option for a fourth year and an opt out after his second season with the club.
  Fact 1.0: According to ESPN's Jeff Passan, the deal is for $42 million.
  Fact 1.1: According to ESPN's Jeff Passan, the deal is over three years.
  Fact 1.2: According to ESPN's Jeff Passan, the deal includes a club option for a fourth year.
  Fact 1.3: According to ESPN's Jeff Passan, the deal includes an opt out after his second season with the club.

Retrieved items:

=== END ITEM id=e8331ec158b98e32 ===

=== ITEM id=00525c73b9671ef6 ===
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

=== END ITEM id=00525c73b9671ef6 ===

=== ITEM id=ba3d22d4daf70968 ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: This person's statement is described as unfortunate.
  Fact 0.1: This person said they had conversations with investors as part of their due diligence.
  Fact 0.2: Alameda was taking on losses from FTX.
  Fact 0.3: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: Huang gave testimony during cross-examination.
  Fact 1.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.2: Huang said Paradigm was told that Bankman-Fried did not want investors as directors.
  Fact 1.3: Huang said Bankman-Fried did plan on having a board with experts, in contrast to not wanting investors as directors.

Retrieved items:

=== END ITEM id=ba3d22d4daf70968 ===
