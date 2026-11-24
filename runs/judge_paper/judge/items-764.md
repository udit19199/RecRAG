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

=== ITEM id=377a66c905f695bb ===
Gold sentences and facts:
Sentence 0: That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.
  Fact 0.0: The answer ultimately lies with the jury.
  Fact 0.1: The jury was selected on Wednesday morning.
  Fact 0.2: The jury was selected before lawyers for the government.
  Fact 0.3: Bankman-Fried swapped two very different stories of the former crypto mogul's sudden rise and almost instantaneous fall.
  Fact 0.4: The former crypto mogul had a sudden rise and an almost instantaneous fall.
Sentence 1: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 1.0: The highly anticipated criminal trial was for Sam Bankman-Fried.
  Fact 1.1: Sam Bankman-Fried is a former CEO of bankrupt crypto exchange FTX.
  Fact 1.2: FTX is a bankrupt crypto exchange.
  Fact 1.3: The criminal trial started on Tuesday.
  Fact 1.4: The criminal trial determines whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 2>

<item 3>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 3>

<item 4>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 4>

<item 5>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I was ready to hear this pursued further — but then it, too, was dropped.

I think the defense was also trying to suggest that the government had coerced Ellison’s testimony, by suggesting that she had pleaded guilty to a charge of
defrauding investors that she couldn’t have been involved in. After all, she didn’t prepare materials for them. Unfortunately, she did say that she had
conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine. This
line of questioning felt like a waste of time.

There were rather a lot of sidebars during the cross-examination, to the point that when one occurred, several jurors looked entertained. There were a few
yesterday, too, including one in which the prosecution complained that Bankman-Fried was visibly scoffing at Ellison’s answers, according to the court
transcript.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Follow along for all the latest news and regular updates from the trial.

Yesterday, during an evidentiary hearing, Bankman-Fried was repeatedly scolded by Judge Lewis Kaplan for not answering prosecutor Danielle Sassoon’s questions
on cross-examination. Today, Bankman-Fried was scolded by Kaplan for not answering his counsel Mark Cohen’s questions on direct examination. Bankman-Fried has
also occasionally interrupted Cohen with “yes” and “yup.”

We spent a lot of the morning explaining vocabulary. I will spare you the full list, though I will say that explaining “Amazon Web Services” and “database” was
a bit too detail-oriented. Then Bankman-Fried tried to define “market manipulation.” After Bankman-Fried gave his definition, Kaplan told the jury that he was
the final authority on that, thanks.

In fairness to Bankman-Fried, he has been clearer and much easier to understand than he was at the evidentiary hearing yesterday. There was a minimum of word
salad today.
</item 6>

<item 7>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Follow along for all the latest news and regular updates from the trial.

But a lot of testimony relied on Ellison recounting conversations she’d had in person, or on auto-deleting text messaging platforms. This gave the defense an
opportunity to try to make her sound unreliable. After all, she had an incentive to flip on Bankman-Fried: the possibility of leniency in her sentencing. Given
her fun tweets about speed, the fact that she was Bankman-Fried’s ex-girlfriend, and that she’d apparently written a bunch of stuff down, I was expecting
fireworks. For the first time in this trial, maybe the defense had an opening.

Instead, I got a sad trombone. In Cohen’s disorganized cross-examination, he mostly bored the jury. At one point, two different jurors appeared to be asleep.

Midway through the morning I began wondering if there was a mercy rule for cross-examinations. Prosecutor Danielle Sassoon had run an effective direct
examination, creating an easy-to-follow narrative.
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“I was concerned that I had unwittingly written code that contributed to a crime,” he said.

Soon, however, the clock neared 4:30 p.m., and court adjourned for the day. Yedidia will continue his testimony on Thursday, followed by Matt Huang, a former
partner at the high-powered venture capital firm Sequoia Capital, and then Gary Wang, a key Bankman-Fried lieutenant and one of the government’s star witnesses.
</item 8>

<item 9>
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
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
In it, Bankman-Fried tried to justify the missing funds through the margin loan program. It was such a flimsy excuse that even George Stephanopoulos could see
through it.

Did Bankman-Fried really think he could get away with it all? He must still think so, or he would not be dragging his closest companions’ names through the mud
in a jury trial. Throughout the entire prosecution’s case, the defense has struggled to shake the credibility of witnesses such as Yedidia, Wang, and Ellison.
The hits the defense made on Singh didn’t make me doubt the overall truth of his story — they just made me think that Singh was kind of self-serving and trying
to secure himself a lenient sentence.
</item 10>

<item 11>
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
</item 11>

<item 12>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 12>

<item 13>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
It is honestly kind of incredible to watch a man torpedo his own credibility on direct testimony. We’re not even at the cross yet, and the judge has already
instructed him to answer the question he’s being asked by his own lawyer.

The jury is watching all of this intently.

The main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his own voice. So far, the count of “Objection,
narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.

Also, sometimes when Bankman-Fried says “we,” he only means himself.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including charges of wire fraud. FTX was a fraud
“from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer
funds.” Follow along for all the latest news and regular updates from the trial.
</item 13>

<item 14>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.
</item 14>

<item 15>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 15>

<item 16>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 16>

<item 17>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 17>

<item 18>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 18>

<item 19>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 19>

<item 20>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 20>
=== END ITEM id=377a66c905f695bb ===

=== ITEM id=96543045fe0fc359 ===
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
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
The Athletic's Ken Rosenthal reported Gurriel has re-signed a contract with the Diamondbacks.

According to ESPN's Jeff Passan, the deal is for $42 million over three years with a club option for a fourth year and an opt out after his second season with
the club.

Yamamoto dines with Mets owner

Date: Sunday, Dec. 17

Source: Joel Sherman, New York Post

The Mets continued their strong push for the top free agent available, Yoshinobu Yamamoto, by hosting the Japanese pitcher for dinner Saturday night at team
owner Steve Cohen's Connecticut home. Cohen and other top Mets officials previously flew to Japan to meet with Yamamoto on his home turf, so this is at least
the second time the key players have met face-to-face.

Sherman reports the bidding process for the 25-year-old "is supposed to intensify this week," with the Yankees, Dodgers, Giants, Red Sox, Phillies and Blue Jays
also in the mix.
</item 1>

<item 2>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov. 17

Source: Daniel Kramer, MLB.com

Here’s the key piece of the article: “Industry sources told MLB.com this week that landing Ohtani doesn’t appear to be within the Mariners’ realistic agenda
this offseason.”

In a way, that’s surprising. The Mariners seem to check a lot of boxes for Ohtani. They’re set up to win for a long time with a bunch of young starting pitchers
and a co-superstar in Julio Rodriguez. Plus, Ohtani would get to be part of the quest for the franchise’s first World Series title, he’d get to wear the same
uniform as his baseball hero, Ichiro, and he’s lived there in the offseason. So it must be about the money, and Seattle’s reluctance to even get in the
conversation. That’s gotta be super disappointing for Mariners fans starved for a title, right?
</item 2>

<item 3>
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
</item 3>

<item 4>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Morosi reports Shohei Ohtani is "likely to decide on a team within the next week," according to a source.

Status quo on Yankees' Juan Soto trade talks

Date: Sunday, Dec. 3

Source: Jon Heyman, New York Post

The Yankees reportedly are pushing hard to acquire Juan Soto from the Padres, but Heyman says the teams haven't spoken since San Diego requested a six-player
package in return for Soto and Trent Grisham. That ask included pitchers Michael King and Drew Thorpe, who the Yankees aren't inclined to move. Stay tuned...

Shohei Ohtani field growing smaller

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

It has been expected since before he hit free agency that Ohtani would set the record for the largest fully guaranteed contract in North American sports
history, with Passan reporting the deal could reach as much as $600 million.
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Verdugo had been the Red Sox's starting outfielder for the last four seasons, coming over as the centerpiece in the trade that sent Mookie Betts to the Dodgers.
Verdugo hit .281 with 43 home runs, 206 RBI and 16 stolen bases over 2,071 plate appearances with Boston. He also led the American League in outfield assists in
2020 and 2023.

In return, Boston will get Weissert and two pitching prospects headlined by Fitts — the Yankees' No. 4-ranked pitching prospect and 12-ranked prospect overall.

Relievers Yates, Devenski get one-year deals for 2024

Date: Tuesday, Dec. 5

Source: Ken Rosenthal, The Athletic and Jeff Passan, ESPN

Veteran relievers Kirby Yates and Chris Devenski know where they'll be pitching in 2024, according to reports Tuesday.
</item 8>

<item 9>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
While the offseason has gotten rolling, trade season hasn’t quite arrived. The Tigers acquired Mark Canha from Milwaukee in the biggest swap of the first couple
weeks. Otherwise, the GM Meetings simply served to set the stage for future transactions.

That’ll surely involve some notable trades. Unlike the summer deadline, projecting which players will change hands is difficult at this stage of the offseason.

Aside from the A’s, there aren’t any teams clearly in rebuild mode. Clubs like the Rockies, Royals, White Sox, Nationals and Angels have uphill paths to
contention, yet none has indicated they’re prepared to punt on the 2024 season entirely.

The Padres, Twins, Rays, Brewers and Guardians are all aiming to contend but have questions about their payroll. That’s standard operating procedure for Tampa
Bay, Milwaukee and Cleveland. San Diego and, to a lesser extent, Minnesota have shown a willingness to spend in recent years but are scaling back — in both
cases, likely tied to their uncertain local television rights contracts.
</item 9>

<item 10>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
20

Source: Ken Rosenthal, The Athletic

Buried at the bottom of Rosenthal’s piece on what an Ohtani contract might look like — incentives, opt-outs and all sorts of stuff — is this little bit of
reporting:

An agreement at the winter meetings in Nashville the first week of December would be the ideal outcome for MLB, but not necessarily for Ohtani. He perhaps will
be better off waiting for the signing of Japanese righty Yoshinobu Yamamoto, who is expected to be posted for free agency as soon as Monday, triggering a 45-day
window for teams to negotiate with him. Losing out on the 25-year-old Yamamoto, who could command more than $200 million, might put certain teams on tilt,
improving the landscape for Ohtani.

Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov.
</item 10>

<item 11>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Bellinger would be returning to the Cubs, the team with which he rebuilt his value after a down year with the Dodgers. If he signed with the Blue Jays, he'd be
bringing the powerful left-handed bat the team sought when it pursued Ohtani.

Dodgers send stars to meet with Yamamoto

Date: Wednesday, Dec. 13

Source: Fabian Ardaya, The Athletic

The Dodgers added one Japanese star this offseason. They're now looking to add another. Los Angeles reportedly sent Ohtani, Mookie Betts, Freddie Freeman and
Will Smith to meet with Yamamoto, trying to convince the standout starting pitcher to join the National League superpower.

Signing Yamamoto would be a major boost for the Dodgers, who still have a need at starting pitcher.

Rays-Dodgers working on a Tyler Glasnow trade

Date: Wednesday, Dec.
</item 11>

<item 12>
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
</item 12>

<item 13>
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
</item 13>

<item 14>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
That informs the offseason trade landscape, which is headlined by a few stars deep into their arbitration windows. Without many clear rebuilders, it’s possible
we see more swaps of big leaguers at positions of need between teams that expect to compete.

Before we get to the list, a note on methodology. This isn’t a strict ranking of players’ trade value, nor is it solely about likelihood of being moved. We’re
trying to balance both of those things, an inherently subjective exercise.

All projected salaries for arbitration-eligible players are courtesy of MLBTR contributor Matt Swartz.

1. Juan Soto, LF, Padres

Other than Shohei Ohtani’s free agency, there may be no bigger story of the offseason than Soto’s future in San Diego. He is entering his final year of club
control, projected for a $33M salary that would be the highest ever for an arbitration-eligible player.

The Padres are scaling back payroll, perhaps as much as $50M.
</item 14>

<item 15>
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
</item 15>

<item 16>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Rays, Mariners talking trades in winter meetings

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

The Mariners and Rays have been frequent trade partners in recent years, and there's a chance they could be talking shop again at the winter meetings. MLB
Network's Jon Morosi reported the teams are discussing Rays utility infielder Isaac Paredes, and potentially star left fielder Randy Arozarena, in talks to
start the winter meetings.

Jays unlikely to land both Ohtani, Soto

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

The Blue Jays have been among the most active names in both the Ohtani and Soto markets. And of course, those talks have had fans dreaming of a lineup that
features, Ohtani, Soto, Vladimir Guerrero Jr., Bo Bichette and George Springer.
</item 16>

<item 17>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
And he is clearly looking for a top-of-the-market contract. Rogers reported sources indicate Bellinger's agent, Scott Boras, is seeking a deal worth well over
$200 million for Bellinger, with one source saying he is "not budging."

Rogers went on to note that some believe the Cubs won't be a team willing to pay a player that type of money until he's 40 years old, while a team like the Blue
Jays might given their desperation to round out the roster.

Padres closing in on contract for Yuki Matsui

Date: Tuesday, Dec. 18

Source: Yuki Yamada, Sankei Sports

The Padres are nearing a deal for one of the top Japanese pitchers on the market. Per Sankei Sports' Yuki Yamada, the Padres are "close to signing" Rakuten
Golden Eagles' reliever Yuki Matsui to a deal.
</item 17>

<item 18>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
15. Randy Arozarena, LF, Rays

Arozarena hasn’t been the subject of the same level of trade speculation as Glasnow has. There’s an argument the Rays should be more willing to move the
All-Star outfielder if they’re looking to create payroll room.

Tampa Bay has greater depth in the outfield than they do on their injury-riddled pitching staff. Josh Lowe, Jose Siri, Harold Ramírez, Luke Raley and Manuel
Margot are all on hand. (Margot has come up in trade rumors as well.)

None of those players is as good as Arozarena. Trading star players for younger talent is familiar territory for the Tampa Bay front office though. Arozarena is
projected for a $9M arbitration salary and controllable for three seasons. He’d have significant trade value if the Rays were to consider moving him.

16. Mariners SP

We initially had Bryan Woo in this spot.
</item 18>

<item 19>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
MLB's winter meetings are always a prime time for moves to happen as many of the top executives, players and agents all come together at the same spot.

Last year's winter meetings wound up being explosive. Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.

This year's winter meetings could wind up being similarly busy. There is a growing sense Shohei Ohtani could sign during the winter meetings, and a move of that
magnitude could kick off the next wave of moves. There's also a number of top players on the trade market, including star Padres outfielder Juan Soto.

Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.

WINTER MEETINGS: Shohei Ohtani rumors, Juan Soto trade destination come to forefront

Sporting News will be tracking all the latest news and rumors from the 2023 Winter Meetings.
</item 19>

<item 20>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Who will hand him that contract has been much more of a mystery. Little has been leaked about the top suitors for Ohtani beyond the usual suspects. Passan
reported that at least three teams initially believed to be in the running — the Mets, Rangers and Red Sox — are believed to be out now. The teams most likely
to still be in the mix are the Dodgers, Cubs, Blue Jays and Angels, with the Giants' status in the race for Ohtani unknown.

Star players could be traded

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

The free agency pool outside of Ohtani, Cody Bellinger, Blake Snell and Yoshinobu Yamamoto is a bit lacking in star power, but the trade market could ensure
several big-name players are on the move this offseason.
</item 20>
=== END ITEM id=96543045fe0fc359 ===

=== ITEM id=51aceb972a1dd9f2 ===
Gold sentences and facts:
Sentence 0: When he entered free agency, a dozen teams lined up, curious to see if they could meet his eye.
  Fact 0.0: A player entered free agency.
  Fact 0.1: A dozen teams lined up when that player entered free agency.
  Fact 0.2: The teams were curious to see if they could meet that player's eye.
Sentence 1: As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and determining the fate of those around me.
  Fact 1.0: The speaker is the player.
  Fact 1.1: The player explores a system of the Imperium.
  Fact 1.2: The player has a loyal crew of companions.
  Fact 1.3: The player makes impactful choices.
  Fact 1.4: The player determines the fate of those around the player.

Retrieved items:
<item 1>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The second, just launched, is the wild Season of Discovery, which remixes and restructures the original WoW experience — interpolating staggered level caps,
shuffling class roles — in ways that might just change MMO design forever. WoW Classic is quietly, and paradoxically, where Blizzard is doing its most
forward-thinking work right now. —Oli Welsh

39. Fire Emblem Engage

Developer: Intelligent Systems

Where to play: Nintendo Switch

Fire Emblem Engage was designed for a very specific kind of sicko: one not particularly interested in the origin stories of a horde of teenagers, or the
politics of a bourgeoise academy, or what kind of tea a teacher prefers, but instead one obsessed with the endless minutiae of combat stats, weapon loadouts,
and team composition. I know this because I am one such sicko.

If you’ve read any of my reviews or essays on Polygon, then you know I prefer strategy games that can get out of their own way.
</item 1>

<item 2>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Those two would be the biggest supporting bats in the lineup around Ohtani. However, both are free agents after the 2025 season, and given the added payroll of
signing Ohtani, it's possible Toronto would view keeping the other two phenoms as too pricey.
</item 2>

<item 3>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
The victory tasted like ash in my mouth after all that frustration.

Voidship combat is another aspect that feels clunky and frustrating. Like the squad-based skirmishes, naval encounters are also turn-based, where positioning is
ultra important. Space naval battles should feel tense, but instead, I’m mostly annoyed at having to continually rotate my ship and set up my zones of attack. I
wish I could delegate these annoyances to my Senechal — to delegate the duties of character leveling and ship combat, the better to appreciate all of the things
Rogue Trader is doing so well.

Rogue Trader is a dense, vast game, and much of it has clearly been crafted with love for the expansive lore of the 40K canon. While there are small annoyances
and clunky features along the way, the political intrigue, cast of characters, and moral choices have me hooked.
</item 3>

<item 4>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
The Giants are considered to be an unknown.

In addition to the teams, Passan also reported Ohtani is expecting a contract somewhere between $550 and $600 million, which would be the largest guaranteed
contract in North American sports.

Blue Jays could alter roster with Ohtani signing

Date: Nov. 30

Source: Jon Heyman, New York Post

The Blue Jays are one of the teams believed to still be among the strongest in the mix for Ohtani. The New York Post's Jon Heyman reported there could be some
changes if Ohtani signs with the Blue Jays, however, reporting that some rivals believe Bo Bichette and Vladimir Guerrero Jr. could hit the trade block if
Toronto lands the star.

Trading the two infielders might seem counterproductive to signing Ohtani in the first place. Those two would be the biggest supporting bats in the lineup
around Ohtani.
</item 4>

<item 5>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
After more than a decade of going on and on about the “board gaming renaissance,” it’s safe to say that tabletop role-play is finally having its moment in the
spotlight. But while Dungeons & Dragons’ OGL fiasco has clearly lit a fire under many players eager to make a change, the fact of the matter is that a sizeable
cohort of writers and designers have been toiling away at excellent games for years now. Their tireless efforts have yielded a bumper crop of excellent, some
might say genre-defining, TTRPGs. Just like board games, a few modern classics have also cropped up: Look no further than The Quiet Year and The One Ring, both
making their second appearance on this list.

Just as before, Polygon asked nearly two dozen writers, designers, presenters, actors, and personalities from around the world of tabletop gaming to share with
our readers the TTRPGs that made an impact on them and their players this year. Here’s what we found.
</item 5>

<item 6>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
For me, Project ECCO found that balance.

A story of time travel and cosmic horror, Project ECCO is easy to get into and relatively quick to play (three to 10 hours per game). Between the hundreds of
narrative prompts, unlocking various time travel “devices” (coins, dice, tarot cards, etc.), and the absolutely stunning layout, this is one of my favorite
TTRPGs, period.

Played in the pages of a yearly planner, Project ECCO hits every beat of a good time-travel story: time loops and timeline divergences, a shady and controlling
Time Travel Agency, and an unknowable cosmic Entity consuming spacetime. As you play, you (literally) burn through the days of your planner, creating an
artifact, marked and changed by the telling of your agent’s story.

For an example of gameplay and a masterfully produced actual play, listen to the Project ECCO miniseason of My First Dungeon with game designer Elliot Davis.
</item 6>

<item 7>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
11

Source: Fabian Ardaya, The Athletic

According to The Athletic, Ohtani will defer a whopping $680 million of his $700 million contract from the Dodgers, receiving only $2 million per year.

While it's important to note the deal will still count as $46 million annually toward the luxury tax thresholds for Los Angeles, Ohtani is still giving the
organization much more flexibility than expected over the course of the deal.

The two-time MVP will receive $68 million per year, without interest, from 2034-43, making the present-day value of the contract much less than it originally
appeared.

MORE: Complete breakdown of Shohei Ohtani's Dodgers contract, by the minute

Giants reportedly meet with Yamamoto

Date: Monday, Dec. 11

Source: Buster Olney, ESPN

ESPN's Buster Olney reports the Giants met with Japanese ace Yoshinobu Yamamoto on Sunday, Dec. 10.
</item 7>

<item 8>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And if you just enjoy a good yarn and have access to basically any screen and $15, then you’re a perfect mark too. It runs as well on console and PC as it does
on iOS and Android, so don’t fret about where you play, just do so and soon! Before Square Enix stops investing in all these oddities. —Chris Plante

22. Marvel’s Midnight Suns

Developer: Firaxis Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

[Ed. note: Marvel’s Midnight Suns was released in 2022, but it just barely missed the cutoff for our best video games of 2022 list, so it’s eligible for our
2023 awards.]

I know what you’re thinking: Another licensed Marvel game? Come on, right? But hear me out. I played Marvel’s Avengers, too, and this isn’t that.
</item 8>

<item 9>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Much like humanity itself? —ML

23. Paranormasight: The Seven Mysteries of Honjo

Developer: Square Enix

Where to play: Android, iOS, Nintendo Switch, and Windows PC

Square Enix has plenty of mega-franchises to fill its time (and its coffers). This year, we have new entries for Octopath Traveler and Final Fantasy, along with
new Dragon Quest and Kingdom Hearts games in the not-so-distant future. Dayenu!

And yet, the publisher can’t help itself from bombarding us with surprising, interesting, sometimes great, often good-enough experiments. In 2022, we got an
English-language remake of lost gem Live A Live, the surprisingly enjoyable tactical RPG DioField Chronicle, a bonkers Final Fantasy spinoff featuring the
musical stylings of Limp Bizkit, and a pair of oddball card games lathered in lore from gaming’s best weirdo.
</item 9>

<item 10>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
You play as a broody Clive Rosfield, a young man whose life’s work is to protect his little brother, Joshua. The story begins when Clive’s life takes a turn for
the worse and he vows to destroy the monster who ruined his and his family’s legacy.

Developed by Creative Business Unit III, Square Enix’s internal team behind the MMORPG Final Fantasy 14, 16 leans into patchwork territories of fantasy genre
fare. There is palace intrigue, a whole lot of sex, and endless war between nations. But the developers then sprinkle in Final Fantasy elements like mother
crystals, dazzling kaiju fights between summons (known as Eikons in this iteration), and of course, Chocobos.

The quality of the story in this long and linear character-driven RPG waxes and wanes, but the action combat is among the best I’ve ever played. The gameplay
grips you from the very beginning as Clive smoothly dashes, parries, and swings his giant sword and varied magic with a dazzling amount of style.
</item 10>

<item 11>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Players take on the role of soldiers and other hangers-on just trying to make it out alive. The heady mix of dice-driven gunplay and a playing-card-style
encounter creation makes it easy to run. But the fun begins even before that with character generation, which uses old school randomized tables to simulate an
entire military career. If you’re looking to get your video gaming buddies around the table for something that doesn’t require a headset or shouting down
pre-teens in a glitchy public lobby, the all-inclusive starter set comes highly recommended.

Project ECCO

It’s a near-impossible task to get a consistent group together, so solo TTRPGs have become my primary method of gaming. However, the barrier for me is finding a
game with a balance of narrative and mechanics that isn’t overwhelmingly crunchy, but also isn’t open-ended to the point of choice paralysis. For me, Project
ECCO found that balance.
</item 11>

<item 12>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
'Money's great, but winning's better'

Date: Nov. 21

Source: Max Scherzer, Rangers pitcher on Foul Territory

OK, so this isn't necessarily a rumor, but an example of how much players would love for Ohtani to join their team. Scherzer was a guest on Foul Territory on
Monday.

"Money's great, but winning's better."



Max Scherzer shares his pitch for Shohei Ohtani to join the @Rangers 🏆



▶️ https://t.co/6KoSRqSgk2 pic.twitter.com/PHW2jHWj6C — Foul Territory (@FoulTerritoryTV) November 20, 2023

Maybe Ohtani won’t sign quite so soon after all

Date: Nov.
</item 12>

<item 13>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
Dogmatic choices exult the God-Emperor and loathe the mutant and the witch, the Heretical options pursue corruptive power, and the Ionoclast path is the closest
thing we have to modern-day “good guy” morality.

After the events of the tutorial, my beautiful voidship is run down and my staff is struggling. As the newly anointed Rogue Trader, I have to trek around the
various planets of the Koronus Expanse to get a new Navigator, fix my ship up, and avoid any major diplomatic incidents with the locals. My voidship is the size
of a modern city; I’m as much a governor as a captain, and I have to manage the ship, its cargo, and its many occupants.

In the process of getting back on my feet, I uncover a nefarious cult and a deep conspiracy. It’s not an easy job, but I’m blessed with a handful of companions
from the Imperium to help me out.
</item 13>

<item 14>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
You can even wear them as mourning jewelry.

This Pathfinder 2nd edition-compatible setting has been as enchantingly addictive as the fictional ichor that propelled Ourichor into the renaissance era that
the project is named after. While there are only two exquisitely illustrated supplements on creatures and equipment available currently, I recommend those with
a love of the unusual to keep an eye out for more in the years to come, such as the upcoming Paragons of Luminant Age that introduces NPCs and explores their
motives through the four moons.

Marvel Multiverse Role-playing Game

As a role-player, I’ve spent a lot of time looking for a good lightweight superhero RPG. I’ve tried quite a few but had yet to find one that really hit the mark
for me until I tried Marvel Multiverse Role-playing Game. I have to say, I was initially dubious about this game but it has blown me out of the water.
</item 14>

<item 15>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: J.P. Hoornstra, Dodgers Nation

The Blue Jays have reportedly won the sweepstakes for Ohtani. According to Dodgers Nation's J.P. Hoornstra, Ohtani has selected the Blue Jays. The official
details have not been reported, though the report indicated it will shatter Mike Trout's record 12-year, $426.5 million deal.

However, Sportsnet's Ben Nicholson-Smith reported Ohtani has not yet made his decision, and there is not a timeline, though he noted Toronto is a finalist.

Decision from Ohtani 'imminent'

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

The wait could soon be over. MLB Network's Jon Morosi reported Ohtani is expected to make his decision soon, reporting it could come as early as Friday.
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec.
</item 15>

<item 16>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
At its best, you really can feel the Force around you. —CA

31. Connections

Developer: The New York Times

Where to play: Android, browser, and iOS

After the acquisition of viral hit Wordle, The New York Times’ Crossword ecosystem leveled up to full-fledged attention competitor — NYT is a media company with
a gaming platform. And its staff of puzzle writers and editors are not stooping to mind-numbing mobile content to keep the expansion going. Connections, its
latest title to pair well with morning coffee, is one of the year’s best games.

Connections offers you a grid of 16 words and a mission: Detect the common threads between four different sets of words without embarrassing yourself. The
grouping logic ranges from simple (“animals”) to silly (“synonyms for farting”) to sneaky (“countries when the letter ‘A’ is added”).
</item 16>

<item 17>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: Bob Nightengale, USA Today; Ben Nicholson-Smith, Sportsnet

Ohtani had a change of plans, apparently. Hours after MLB Network's Jon Morosi reported that the two-way star was headed to Toronto for a meeting on Friday, USA
Today's Nightengale and Sportsnet's Nicholson-Smith reported that Ohtani is at his home in Southern California.

The Blue Jays are still finalists to land Ohtani but his landing spot is not official yet.

Ohtani headed to Toronto on Friday

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

Ohtani might not have officially picked a team, but he is headed to Toronto, per MLB Network's Jon Morosi. It had been previously reported he had picked the
Blue Jays, though several reports have since denied that he has made a decision.

Report: Ohtani picks Blue Jays over Dodgers

Date: Friday, Dec. 8

Source: J.P.
</item 17>

<item 18>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
For 40K fans, this is a rare treat — a game that digs past the heroic facade of bolters and battles and taps into the grimdark dystopia that makes this
particular sci-fi setting so damn compelling.

Warhammer 40,000: Rogue Trader will be released on Dec. 7 on Windows PC. The game was reviewed using a pre-release download code provided by Owlcat Games. Vox
Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links.
You can find additional information about Polygon’s ethics policy here .
</item 18>

<item 19>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The best games this year told their stories in interesting ways, be it ignoring your superpowers to bike through your old neighborhood in Queens, a musical
number, or fractured memories hidden in a puddle. But Baldur’s Gate 3 takes experimental storytelling to another level. Every choice and every new direction is
a simple success, fail, or crit away. One roll of the dice after another can take you all the way from a crashed Nautiloid to the city of Baldur’s Gate itself.

In a year filled with beautiful stories and powerful moments, it’s the spinning whir of Baldur’s Gate 3’s d20 that defined video game storytelling in 2023. —RG

1. The Legend of Zelda: Tears of the Kingdom

Developer: Nintendo EPD

Where to play: Nintendo Switch

Princess Zelda is missing. Ganondorf has returned, somehow, and he’s really hot. So an age-old hero in a green tunic has to step in.
</item 19>

<item 20>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I can call on them to unlock doors or perform other environmental checks, but they come most in handy in combat.

When I meet opposition, it’s deeply satisfying to control my troops in a turn-based battle. Each fight takes place on a grid; it’s very similar to Baldur’s Gate
3 or even XCOM 2. Some positions provide cover, while others are out in the open. Friendly fire is also a very real concern. An arc of auto-fire from a bolter,
or a Navigator’s third eye opening, can harm friend as well as foe. My Rogue Trader is a sniper, and she would be lost without her Senechal taking the front
lines. There’s a lot of firepower at my disposal, and it’s mostly quite satisfying to use — even if I occasionally shred my poor Senechal with a devastating
AOE.

I can chat with the companions between battles, learning more about their pasts.
</item 20>

<item 21>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The Quiet Year

Em Friedman, associate professor of English at Auburn University and Polygon contributor

Look, The Quiet Year appeared not once but twice on last year’s roundup, I know. Jay Dragon praised its elegance and Keerthi Sridharan noted the way their game
group used it as a prologue (as Friends at the Table and The Adventure Zone have popularized), and it’s a standard early on in my TTPRG class. But for me it was
the standout game of this year because after running it for my niece (then 4) and nephew (8), my nephew asked that I run “the skulls game” for his entire class.
While not intended for a dozen grade schoolers, it turns out the mechanics of The Quiet Year — which ensure everyone gets to have their turn unimpeded by the
commentary of others, ritualizes dissent and discussion, as well as having a glorious sheet of blank paper to draw the zombie chicken monster friend of your
dreams — turned out to be a perfect framework to imagine how to work together.
</item 21>

<item 22>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 22>

<item 23>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
What difference does it make if something scores highly on some arbitrary rubric if it doesn’t resonate with the people in your community? And so I look at
year-end lists like these as a kind of menu, boards of fare that clever game masters can use to pick just the right flavor for a given table.

For me this year was all about reconnecting with the other dads in my neighborhood after years spent locked inside during the pandemic. I spun up a Discord that
we used to organize friendly games of Call of Duty’s DMZ mode, organized a few visits to the local brewery to catch up make new connections, and generally just
tried to get awkward, middle-aged men accustomed to seeing each other socially again. One night I decided to bust out Twilight: 2000, and it was an instant hit.

Twilight: 2000 is a game about surviving in the aftermath of a fictional nuclear conflict in the heart of Europe. It’s an alternate history where the Cold War
went hot, swallowing up an entire United States Marine division.
</item 23>

<item 24>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I love the characters, the environments, the writing, the lore, and the flow of battles. But I have concerns with the game’s pacing. By the end of the first
chapter, I had leveled up 16 times. Each level offered marginal rewards, like being able to move slightly farther during the character’s turn in combat or
having a higher parry chance when being attacked. A slow drip-feed system means each level feels less important, and even though I’m growing stronger, I don’t
get that sense of long-term satisfaction.

It’s especially frustrating to hit a roadblock like the one at the end of Act 1, where an incredibly tough boss rolls out of nowhere and spanks my crew — and I
can’t leave to go grind experience somewhere else. I eventually found out a way to cheese the fight by focusing on my melee fighters’ positioning, but it took
far too long banging my head against the wall. The victory tasted like ash in my mouth after all that frustration.
</item 24>

<item 25>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
That’s gotta be super disappointing for Mariners fans starved for a title, right?

On the other hand, maybe the M’s know all about Ohtani’s desire to keep the rumor mill quiet, and leaking “news” like this would please Ohtani and his camp.

‘Timing could be right for a Cubs-Ohtani union’

Date: Nov. 16

Source: Jesse Rogers, ESPN

The Cubs were one of the seven finalists for Ohtani the first time around, but lack of the DH in the NL at the time made it hard for Ohtani to choose Chicago.
His primary goal was showing he could both hit and pitch at a high level for an MLB team, and trying to play the outfield instead of just DHing would have been
too much, even for Ohtani.

Now, though? The NL has the DH, and the Cubs are itching to make a sport-shaking signing. Well, another sport-shaking signing.
</item 25>

<item 26>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
There are so many puzzles in this game — there’s even a puzzle metagame, spread across the game’s gorgeous map — and they’re each excellent, teaching you new
concepts before refracting them and forcing you to think differently. —Nicole Clark

Related November games you might have missed

44. Fading Afternoon

Developer: yeo

Where to play: Windows PC

Video games demand an account for your time. Some video games track the minutes and hours you’ve spent on screen. Others impose strict limits on how long you
play, or how much you can do. But all video games constantly ask: How will you spend your time? Seiji Maruyama doesn’t know how much time he has left. Fresh out
of prison, he’s not a young man anymore, but he doesn’t want the streets to know. So he returns to his life as a yakuza heavy, hoping to make his mark again.

Fading Afternoon ends when Seiji’s time runs out.
</item 26>

<item 27>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Here’s what we found.

City of Winter

Linda Codega, Ennie Award-winning entertainment journalist

My friends and I still talk about the weekend we spent playing City of Winter. With lightweight rules but a heavy emphasis on the kind of ephemera usually
reserved for much crunchier games, creating space for this game is almost as important as who you chose to play it with. Characters travel along a cloth map
that illustrates a landscape with only barest indications of topography; the focus instead relies on themes of assimilation, diaspora, and — most importantly to
the narrative — baggage. As a family of immigrants (bound by blood or by choice), players describe their home, their traditions, and then, as a darkness comes
from the west side of the map, they leave it all behind, choosing what traditions and rituals they pick up from the places they visit and what they hold on to
from their homeland.
</item 27>

<item 28>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
But the fresh grand dame of my tabletop world is Triangle Agency, a gleaming, razor-sharp new TTRPG that feels like a cross between The X-Files, Annihilation,
and playing Control with the Robert Anton Wilson mod turned on.

In Triangle Agency, you are super-abled agents investigating and stopping ontological anomalies from destroying reality, with mechanics that feel like the
platonic ideal mix of analytic crunch and grand narrative weirdness. And my god, it is gorgeous, with jaw-dropping art and design that puts it on the same shelf
as Wanderhome and Mörk Borg. The full game ships early in 2024, but you can get the digital rules and start playing right now, for the low, low price of
“whatever you think is appropriate.”

Twilight: 2000

Charlie Hall, senior editor, tabletop

I firmly believe that tabletop role-play should always take into account the audience that sits down at the table to play. That’s part of the reason why I hate
best-of lists for this particular category of games in particular.
</item 28>

<item 29>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I create my custom protagonist, determining stats and bonuses based on my backstory, home world, and so on. I’m then thrust into a high-pressure job interview
as a potential heir to a Rogue Trader aboard her flagship. Little do we know that there’s a coup in the works, and traitors aboard the ship. After a deadly
struggle against heretics and demons, I ascend to the position of Rogue Trader. I’m the captain now, and I get to decide how to run my ship.

This is a vast game, with tons of features you’d expect from a CRPG — companions (each with their own narrative paths and conversation trees), top-down
strategic combat, and branching choices that impact the world around you. I get to make choices all day — in fact, that’s part of the gig of a Rogue Trader.
Some are more important than others; whenever I get to specific points in the story, I can choose from one of three major paths.
</item 29>

<item 30>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
All eyes since the start of the offseason have been on Shohei Ohtani. And for the most part, there hasn't been much to see.

The two-way phenom is the most sought-after free agent in MLB history, with his combination of talent and marketability making him a franchise-changing player.

And the bidding for Ohtani could be winding down soon. Multiple reports have indicated Ohtani could sign at some point during the winter meetings, which are
scheduled to take place from Dec. 3 through Dec. 6.

Many of the top teams are expected to be in the mix for Ohtani. The Dodgers and Cubs are among the top teams expected to be aiming to sign the two-time AL MVP,
and plenty of others should continue to compete through the end to land him.

Here's the latest on Ohtani's free agency pursuits.
</item 30>

<item 31>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
20

Source: Ken Rosenthal, The Athletic

Buried at the bottom of Rosenthal’s piece on what an Ohtani contract might look like — incentives, opt-outs and all sorts of stuff — is this little bit of
reporting:

An agreement at the winter meetings in Nashville the first week of December would be the ideal outcome for MLB, but not necessarily for Ohtani. He perhaps will
be better off waiting for the signing of Japanese righty Yoshinobu Yamamoto, who is expected to be posted for free agency as soon as Monday, triggering a 45-day
window for teams to negotiate with him. Losing out on the 25-year-old Yamamoto, who could command more than $200 million, might put certain teams on tilt,
improving the landscape for Ohtani.

Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov.
</item 31>

<item 32>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Mike Mahardy

How the Polygon top 50 list works

Over the past few weeks, the Polygon staff voted, championed, debated, and ultimately threw up its hands and marveled at the list of mammoths, curiosities,
puzzle boxes, and black holes that is our top 50 games of 2023. Any video games that were released in 2023, received substantial updates in 2023, or achieved
renewed cultural relevance in 2023 were eligible for this list. Last year, the cutoff for consideration was Nov. 30. (You’ll notice a certain Firaxis Games
joint fairly high up our list.) This year, the cutoff was the same. Should we be thoroughly enamored with Warhammer 40,000: Rogue Trader or Avatar: Frontiers of
Pandora, we’ll make sure they’re considered for next year’s top 50.

Top 50

50. Mr. Sun’s Hatbox

Developer: Kenny Sun

Where to play: Nintendo Switch and Windows PC

Mr.
</item 32>

<item 33>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
He added there are believed to be five teams in the running for Ohtani.

Ohtani decision coming this week

Date: Dec. 3

Source: Jon Morosi, MLB Network

With the winter meetings set to begin in Nashville, it seems a resolution is near for Ohtani. MLB Network's Jon Morosi reports the superstar is "likely to
decide on a team within the next week," according to a source.

Ohtani race starting to clear up

Date: Dec. 1

Source: Jeff Passan, ESPN

Two days before the start of the winter meetings, the picture on Ohtani is beginning to clear a bit more. ESPN's Jeff Passan reported the Rangers, Red Sox and
Mets appear to be out of the running for Ohtani, while the Dodgers, Cubs, Blue Jays and Angels remain in the mix. The Giants are considered to be an unknown.
</item 33>

<item 34>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Triangle Agency

“I laughed so hard my face hurt.” —me, 10 minutes after my first session of Triangle Agency

I love a game with style and a clear, fun point of view, so I’ve been a fan of Triangle Agency since I picked up its player guide in beta form earlier this year
(to prep for writing a published adventure for the game). Written in the form of a cheerful, slightly threatening employee handbook, the Field Agent Manual
guides you through the creation of your supernaturally powered character, then gets you ready for your role finding and containing Anomalies that threaten the
nature of reality. After your team’s morning meeting, of course.

Beyond the style of the materials, though, playing the game was an absolute blast. The mix of mundane life, weird occurrences, and corporate BS created a
fast-paced game experience that was playful and easy to understand (just count those threes!
</item 34>

<item 35>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Playing Exquisite Biome feels a little bit like being David Attenborough on an alien planet.

With beautiful, evocative art and short, easy-to-parse rules, Exquisite Biome is my go-to “what if we made up some weird little guys and set them in motion for
an hour” game. And truly, what higher calling is there?

Fetch My Blade

Did you know you can play one of the best tabletop RPGs whenever you want, without any scheduling nightmares or unfeasible time commitments? You can when you
play Fetch My Blade, a solo journaling role-playing game created by Kelly Tran and Ethan Yen!

In Fetch My Blade, you role-play as a friendly or fierce dog of a retired legendary sword master. Create your character by choosing one of six dog breeds,
including shih tzu, corgi, mastiff, border collie, husky, and chihuahua, or select your own! Determine three truths about your master and aid them when a
mysterious stranger challenges them to a duel.
</item 35>

<item 36>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The book itself has a stunning and distinct art style that immediately transports you into the doom metal fantasy of the setting; but that’s only a small part
of its allure.

Unlike many TTRPGs that one merely has aspirations of playing one day, Mörk Borg does everything in its power to make the game easy to run. Although it’s a
short read, no line is wasted. Even item descriptions are dripping with hints at nefarious deeds one must have done to acquire them. The optional classes are an
absolute delight for those of us that savor the idea of playing the unsavory. From the esoteric introverts that choose to make their hovel far from prying eyes
to the unfortunate discarded-at-birth cutthroats who’ve had to scratch out a living any way they can, there is something for everyone. There is always the
option of creating something more tailor-made if none of the classes are appealing, but I appreciate that Mörk Borg offers beginner-friendly ideas and
archetypes to help you create something that fits right into the gritty setting.
</item 36>

<item 37>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec. 8, 2017, six years to the day.

Dave Roberts says Dodgers met with Ohtani

Date: Tuesday, Dec. 5

Source: Alden Gonzalez, ESPN

Dodgers manager Dave Roberts told reporters Tuesday that the team met with Ohtani for 2-3 hours over the weekend. Roberts said Ohtani is "clearly" the Dodgers'
top priority, adding that the team is confident the two-time MVP will pitch again. Roberts wasn't able to offer a timeline for Ohtani's decision.

Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec. 5

Source: Bob Nightengale, USA Today

The Cubs have been viewed as one of Ohtani's likeliest destinations for a while. It appears that might no longer be the case.
</item 37>

<item 38>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
We have been promised very little of that, though. ESPN’s Jeff Passan reported this: “If visits between Ohtani and a team are reported publicly, it will be held
against the team.”

So, yeah. Probably won’t hear a ton, but that won’t stop the speculation. And it won’t stop us from passing along what precious few nuggets that are scattered
about.

MORE: Way-too-early MLB Power Rankings for 2024

Shohei Ohtani free agent rumors and updates

Ohtani has an 'affinity' for Boston, it seems

Date: Nov. 27

Source: Jeff Passan, ESPN

Hope everyone enjoyed the Thanksgiving holiday. Us baseball writers are pleased that Ohtani didn't decide to sign while the turkey was still warm. This isn't
exactly "breaking news" because rumors have been swirling, but it's an aside from an in-the-know reporter that just might give one fan base a little too much
hope and excitement.
</item 38>

<item 39>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The game also sheds its predecessor’s odious Spider-Cop bits in favor of establishing a deeper connection between the Spider-Mans and their city, giving the
game room to breathe while further immersing you in its heroes’ worlds and struggles.

And then there’s the gameplay. For my money, there are few experiences more enjoyable in gaming than swinging around New York City with Spider-Man. It was fun
in the early 2000s, and it’s still fun now. And with a bigger map to explore (and wings to fly, if you so choose to use them), there’s basically no limit to the
fun you can get up to. The deployment of two protagonists is seamless — Peter and Miles play differently, bringing their own stories and desires to the table —
and switching between them is effortless. It’s also the rare open-world game that doesn’t feel bloated with missions and side quests, instead leaving you
wanting more Spider adventures. —PV

7.
</item 39>

<item 40>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It’s a time in your life where you feel so, so much. You can see that earnestness in the teenage experience where everything is a big, huge issue — sometimes to
the point of cringe — tied up in that big, global issue of the meteor that’s looking to destroy everything. —N. Carpenter

Related August games you might have missed

40. World of Warcraft Classic

Developer: Blizzard Entertainment

Where to play: Mac and Windows PC

Something very, very interesting started to happen in the retro version of WoW this year. While Blizzard has been content to march one half of the game’s
community forward through its history of expansions, it’s started to come up with creative ways to keep the other half — the half that wants to stay in the game
as it was at launch — engaged.

The first of these was Hardcore, a brilliant permadeath mode that instantly made this aging game both more dangerous and more social, resurrecting the spirit of
its 2004 servers.
</item 40>

<item 41>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Furthermore, in the process of putting a game together, I found a glut of free online tools from dungeon generators to NPC character sheet generators that made
game prep a breeze. I find myself coming back to this game again and again with each game session only giving me ideas for future campaigns. I know I’m not
alone in this feeling given the number of Mörk Borg-compatible games that have sprung up this year, from Pirate Borg to Chris Lockey’s upcoming Abyss of
Hallucinations.

I’m Sorry Did You Say Street Magic

Madison Durham, writer and journalist, staff writer at Reviewed

To me, some of the most joyful moments in tabletop gaming come from the experience of creative collaboration — the magic of making something new with friends,
be it a narrative, a character, a world, or in this case, a city. I’m Sorry Did You Say Street Magic by Caro Asercion is a GM-less city-building game, played
between a group of two to six people with a stack of index cards and your favorite writing utensils.
</item 41>

<item 42>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In Alan Wake 2, everyone’s story is happening at once, branching out in endless directions, and we might not know the genre until it’s too late. It’s a
hypertext mystery, an ergodic game that molds itself to an audience raised online, though few computers are in sight. It’s a game for a world suspended in an
endless second act — forever wars, live services, franchises, infinite scrolls — endlessly searching for a conclusion. —JR

2. Baldur’s Gate 3

Developer: Larian Studios

Where to play: PlayStation 5 and Windows PC

Baldur’s Gate 3 hit at the perfect moment. Despite the handful of truly great computer role-playing games that have been released since the early 2000s (Larian
Studios itself made waves with its Divinity: Original Sin series in 2014 and 2017), the genre has largely remained a modern niche. But times have changed:
Dungeons & Dragons has undergone something of a renaissance, thanks to actual-play series like Critical Role and Dimension 20.
</item 42>

<item 43>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Galaxy-spanning faction quests give you plenty to do, but it’s the side quests that really make you feel like the captain in something Star Trek adjacent:
generation ships and superhero hideouts and colonies of clones. There’s good sci-fi here for those willing to make the journey.

It’s true that the game oversold the idea of a galaxy with over 1,000 planets, a number that pales in comparison to the game’s most obvious competitor, No Man’s
Sky. It’s a shame this became so much of the focus thanks to Bethesda’s marketing, because that same focus obscured the absolutely fascinating new game plus
mode at the game’s heart. Consider this: No Man’s Sky is the game that realizes the universe’s infinite number of planets, while Starfield realizes the
universe’s infinite number of possibilities. —Clayton Ashley

33.
</item 43>

<item 44>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Imagine a lineup consisting of Ohtani, Ronald Acuna Jr., Ozzie Albies, Austin Riley and Matt Olson?

Sources: Braves are among the teams still engaged with Shohei Ohtani’s camp, as I reported in this @MLBNetwork segment. @MLB pic.twitter.com/lP4lISzttN — Jon
Morosi (@jonmorosi) December 4, 2023

Ohtani has already received several $500+ million offers

Date: Dec. 3

Source: Jon Heyman, NY Post

It's no surprise the bidding for Ohtani is heating up, with the star expected to sign soon. And New York Post's Jon Heyman reported Ohtani has already received
several deals that have topped $500 million, which could mean he winds up being the first $600 million player in MLB history. He added there are believed to be
five teams in the running for Ohtani.
</item 44>

<item 45>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As we donned Samus’ suit and explored strange planets, aggressive alien lifeforms could now get right in our faces, forcing us to dodge, strafe, and roll (in
morph ball form, naturally) using all three dimensions. No longer would we sit back and watch as Samus dipped her toe into a pool of lava; in first-person, as
molten fire spread over our visor, we’d really feel the pressure to find that Varia Suit upgrade. And perhaps most importantly, from behind Samus’ visor, we
gained the ability to scan our enemies and environment, collecting and translating logs from the long-dead Chozo aliens who once inhabited these now-hostile
places.

The world of Prime is harsh and unrelenting. (Save points will, at times, be quite far from one another.) But it’s worth buckling down and pushing through the
pain points to discover this world’s secrets. —M. Myers

18.
</item 45>

<item 46>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
“In the grim darkness of the far future, there is only war.” That’s the tagline of Warhammer 40,000, one of the most over-the-top and brutal sci-fi settings
around. But even in a merciless, brutal dystopia that grinds its people into dust, some get to enjoy being on the top of the food chain. Warhammer 40,000: Rogue
Trader is about the perils and pleasures of being atop that particular hierarchy.

Rogue Traders in 40K are freelance explorers, scouting the far frontiers of the Imperium of Man. It’s a risky career, but it comes with luxurious rewards:
agency, freedom, power, and a giant flagship. As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and
determining the fate of those around me.

Rogue Trader is a computer role-playing game in the vein of Baldur’s Gate 3 or Pillars of Eternity.
</item 46>

<item 47>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
The Hot Stove dominoes have started to fall, but everyone who loves baseball — fans, players, coaches, front-office types and everyone else — is waiting for
free agent Shohei Ohtani’s decision. Where will the global superstar sign his next contract?

He’s played his entire six-year MLB career with the Angels, but his teams haven’t even sniffed the postseason and Ohtani’s made his desire to win crystal clear,
so he won’t be back. Will he chase every dollar possible, going solely by the highest offer? That seems unlikely, too. Remember, Ohtani wants to win.

Lots of teams can meet Ohtani in that middle ground, offering both lots of money — he will absolutely sign the largest contract in baseball history — and lots
of chances to compete for a World Series title (or two or three). It’s a long list.

MORE: Ranking 15 possible destinations for Shoehei Ohtani

So, the baseball world waits for news.
</item 47>

<item 48>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Here's the latest on Ohtani's free agency pursuits.

MORE: Projecting Shohei Ohtani's contract in free agency

Shohei Ohtani news: Latest updates, rumors for MLB free agency

Ohtani agrees to 10-year, $700 million deal with Dodgers

Date: Saturday, Dec. 9

Shohei Ohtani announced Saturday that he has agreed to a contract with the Dodgers. It's a colossal 10-year, $700 million deal with "significant deferrals," so
while Los Angeles won't pay him $70 million per year, the franchise will be paying him beyond those 10 seasons.

While rumors connected Ohtani to the Blue Jays on Friday, it's the Dodgers who hand him a record-shattering contract and pair the two-time MVP with two former
MVPs in Mookie Betts and Freddie Freeman.

Ohtani not headed to Toronto, at his home in Southern California

Date: Friday, Dec.
</item 48>

<item 49>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Morosi reports Shohei Ohtani is "likely to decide on a team within the next week," according to a source.

Status quo on Yankees' Juan Soto trade talks

Date: Sunday, Dec. 3

Source: Jon Heyman, New York Post

The Yankees reportedly are pushing hard to acquire Juan Soto from the Padres, but Heyman says the teams haven't spoken since San Diego requested a six-player
package in return for Soto and Trent Grisham. That ask included pitchers Michael King and Drew Thorpe, who the Yankees aren't inclined to move. Stay tuned...

Shohei Ohtani field growing smaller

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

It has been expected since before he hit free agency that Ohtani would set the record for the largest fully guaranteed contract in North American sports
history, with Passan reporting the deal could reach as much as $600 million.
</item 49>

<item 50>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
He’s cruel, depraved, and an absolute hoot. Yes, he may flay a few too many people for my tastes, but he’s the best companion to bring to a party.

In the grand scheme of things, this is one of the most complete and detailed explorations of the 40K universe you can find. The game is an homage to the
Warhammer 40K RPGs from Fantasy Flight Games, including Rogue Trader and Dark Heresy. I spent years as a teenager and young adult poring over these sourcebooks
(which Rogue Trader is inspired by), learning more about the tiny details of life in the Imperium. Developer Owlcat Games has paid the same attention to every
detail of the Koronus Expanse. I delve into ancient facilities staffed by tech-priests of the Machine God, the hostile xenos city of Commorragh, or massive
cities built to honor the God-Emperor of Mankind.
</item 50>
=== END ITEM id=51aceb972a1dd9f2 ===

=== ITEM id=e3dc0402149e7ffb ===
Gold sentences and facts:
Sentence 0: Not only has he produced at a high level, but his play has been instrumental in the Vikings turning things around after a 1-4 start.
  Fact 0.0: He has produced at a high level.
  Fact 0.1: His play has been instrumental in the Vikings turning things around.
  Fact 0.2: The Vikings had a 1-4 start.
Sentence 1: But they've also seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero completions of 20-plus yards from tight ends).
  Fact 1.0: The team under discussion has seen the fourth-lowest ADOT from tight ends.
  Fact 1.1: The ADOT from tight ends is 5.0 yards per target.
  Fact 1.2: Teams seem to know they will not be able to succeed on downfield shots against the team under discussion.
  Fact 1.3: The team under discussion has had zero completions of 20-plus yards from tight ends.

Retrieved items:
<item 1>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 1>

<item 2>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
10:42 p.m.: A 15-yard facemask penalty on Broncos WR Marvin Mims Jr. backed up the offense and the Vikings' defense held strong again as a result. Here comes
Will Lutz for his fifth field goal attempt of the game.

10:38 p.m.: INTERCEPTION! Joshua Dobbs was hit as he threw and Broncos CB Ja'Quan McMillan came up with the pick. It was LB Josey Jewell who brought the
pressure on Dobbs. Denver takes over from the Vikings' 9-yard line after McMillan picked up nine yards on the interception return.

Vikings 17, Broncos 12

10:34 p.m.: FIELD GOAL — Will Lutz buries his fourth field goal of the day, this time from 37 yards. Lutz has been automatic in this one, going a perfect
4-for-4 to account for all of Denver's points.

10:32 p.m.
</item 2>

<item 3>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Denver punted to Minnesota's 25, where Joshua Dobbs and the offense will take over.

Vikings 7, Broncos 3

8:56 p.m.: TOUCHDOWN! Joshua Dobbs, are you kidding me?! Dobbs scrambled out right and looked like he had nowhere to go. The QB shed a tackle on the sidelines
and regained his composure, finding TE Josh Oliver wide open in the end zone for a touchdown after the play broke down.

End of first quarter: Broncos 3, Vikings 0

8:53 p.m.: The Vikings' run game is marching the offense down the field. Spell RB Ty Chandler has collected 29 total yards on this drive while starting RB
Alexander Mattison has picked up 27 yards. Minnesota is down to Denver's 4-yard line as the first quarter comes to a close.

8:44 p.m.: The Vikings' defense responds with a stop of their own.
</item 3>

<item 4>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
6 and No. 7 seed for the Vikings. The ramifications could be much bigger than that, though. A loss would mean the Packers, Rams and Saints only trail Minnesota
by a half-game in the wild card race, while a win would keep the Vikings well within striking distance in the NFC North.

6:55 p.m. — D'Onta Foreman remains out for the Bears, along with CB Tyrique Stevenson.

6:40 p.m. — There's no understating the importance of these final six games for Justin Fields, who is fighting to prove he's the Bears' franchise quarterback.
Chicago has a strong chance of landing the No. 1 overall pick in April's draft, thanks to the Panthers' ugly season.

6:20 p.m. — The Vikings are officially without Justin Jefferson tonight, as they'll hold out their star receiver until they return from their bye in Week 14.

Vikings officially downgraded WR Justin Jefferson to out for tonight’s game vs. the Bears.
</item 4>

<item 5>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
3 wide receiver in the offense behind Justin Jefferson and K.J. Osborn, but even with the donut Addison still has a five-catch lead on Osborn on the season with
68 more yards gained.

THIS WEEK: You can bank on the Vikings reversing course against the Chiefs. Two of the past three offenses against Kansas City wound up throwing at least 71% of
their snaps. The team that didn't? Chicago, who was so badly blown out by halftime that they ran the ball 55% of their second-half snaps just to get out of
Arrowhead.

CHIEFS: Have been stout against receivers all season long, ranking top-five in catch rate allowed (53.4%) and YAC/reception allowed (2.96), and sixth-best in
yards per catch given up (11.1).
</item 5>

<item 6>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
10:01 p.m.: The Vikings are moving the chains. They've picked up two first downs in five plays, gaining 43 yards in the process. Alexander Mattison picked up 23
yards on the ground, and Joshua Dobbs and Jordan Addison connected on an 18-yard pass to pick up the bulk of the yardage.

9:56 p.m.: A holding penalty derailed the Broncos' drive as they came up way short on a 3rd & 20 attempt. Minnesota will take over from its own 31 after a
19-yard punt return from Brandon Powell.

9:54 p.m.: Denver receives to start the second half and we're back in action.

End of first half: Vikings 10, Broncos 9

9:38 p.m.: FIELD GOAL — Will Lutz from 52 yards... it's good! The Broncos kicker steps up to bury a long field goal as the first half comes to a close.
</item 6>

<item 7>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
And Edwards follows it up with the Bears 3rd INT of the game ‼️



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/2d4C1s2O09 — NFL (@NFL) November 28, 2023

10:06 p.m. — After a hold by Jaylon Johnson gifts the Vikings a first down, they still can't capitalize. Ty Chandler can't make T.J. Edwards miss on third down,
so Minnesota will go for it on fourth down.

10:02 p.m. — The running game is working, the passing game is not for the Vikings. Mattison has 19 yards three plays into this drive for Minnesota.

Bears 6, Vikings 3

9:58 p.m.
</item 7>

<item 8>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
That just hung up there. The Bears have been all over Dobbs and the Vikings' offense early on. A taunting penalty after the play hurts Chicago's field position.

Bears 3, Vikings 0

8:44 p.m. FIELD GOAL — The drive fizzles out, and the Bears opt to kick the field goal on 4th & 3. It's good. The Bears are out to an early 3-0 lead, but
they've left plenty of points on the board.

End of first quarter: Bears 0, Vikings 0

8:39 p.m. — What a play by Fields to get the ball to a wide open Kmet! The Vikings brought intense pressure, allowing Kmet to get wide open and go for 24 yards.
The Bears are in business.

8:38 p.m. — 2nd & 22 becomes 2nd & 27, but the Bears make up 17 yards on two plays.
</item 8>

<item 9>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ CHI MIN -2.5 O/U 44.5 OPP VS WR 20th PROJ PTS 10.7 WR RNK 33rd YTD Stats REC 13 TAR 26 REYDS 166 TD 2 FPTS/G 8.3 No doubt there's
some potential here, but some hunches keep me from falling for Osborn this week. One: I think the Vikings will try to get their run game going since they don't
have Justin Jefferson. Two: Jordan Addison is far more explosive and T.J. Hockenson has been a much more relied-upon weapon. They should have more targets than
Osborn. And the expected zone coverage Chicago will play further will limit Osborn unless he sees more targets than expected.
</item 9>

<item 10>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
8:44 p.m.: The Vikings' defense responds with a stop of their own. A quick three-and-out for Russell Wilson and the Broncos. Minnesota will take over from its
own 35 after a 12-yard punt return from Brandon Powell. Broncos lead 3-0 with 5:01 to play in the first quarter.

8:38 p.m.: Joshua Dobbs had the Vikings' offense moving but the Broncos' defense came up with a big stop on 3rd & 8. Minnesota had picked up 28 yards in six
plays but Dobbs was nearly picked off on third down, forcing a punt. Denver will take over from its own 25-yard line after a touchback.

8:32 p.m.: Joshua Dobbs was evaluated in the medical tent after taking a big hit on the last drive, but he did not miss a snap at QB for the Vikings.

Broncos 3, Vikings 0

8:30 p.m.
</item 10>

<item 11>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
That play call has not worked for the Bears in this half.

Vikings 10, Bears 9

10:43 p.m. TOUCHDOWN — The Vikings turn the Fields fumble into the first touchdown of the night! Hockenson goes up and gets Dobbs' pass for the score. The
Vikings lead, 10-9.

The first TD of the game is a Dobbs dot to Hockenson!



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/b1IiFoqcqo — NFL (@NFL) November 28, 2023

10:41 p.m. — Dobbs hits Hockenson and pushes the Vikings into the red zone. Minnesota is driving. No one has scored a touchdown yet tonight.

10:40 p.m.
</item 11>

<item 12>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.
</item 12>

<item 13>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Justin Jones nearly had it, but it was Kyler Gordon who came away with it. Everything the Vikings have done on the offensive side of the ball tonight looks like
a chore.

10:24 p.m. — A third down catch by Jordan Addison extends the drive for Minnesota. Addison only has 25 yards tonight.

10:23 p.m. — Ty Chandler gets the ball on second down and gets nothing. Chandler has not looked very effective tonight with 11 yards on five touches.

Bears 9, Vikings 3

10:18 p.m. FIELD GOAL — After Santos missed from 48 earlier, he drills it from 55. It's 9-3, Cubs Bears.

10:16 p.m. — Byron Murphy comes in to break up a third down pass by Fields, and Santos will come out for a long 55-yard field. He's already missed from 48
tonight.
</item 13>

<item 14>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.
</item 14>

<item 15>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 15>

<item 16>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
When it mattered most, though, Fields was poised and sealed the Bears' first win over an NFC North opponent since 2021.

While Chicago's hopes are still extraordinarily distant at 4-8, the Vikings saw their playoff chances take a hit on Monday night. Minnesota (6-6) dropped to the
No. 7 seed in the NFC with the loss, with the Packers, Rams and Saints all just a half-game behind in the wild-card race. While a win would have put the Vikings
in position to compete for the division crown down the stretch, just holding onto a playoff spot will now be a battle for Kevin O'Connell's squad.

MORE: Latest updates on Justin Jefferson's return date

The Sporting News tracked live updates from the "Monday Night Football" game between the Bears and Vikings. Check out all of the key moments you might have
missed.

Bears vs.
</item 16>

<item 17>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.
</item 17>

<item 18>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Denver forced a turnover on downs on Minnesota's final drive, extending its win streak to four games while handing the Vikings their first loss in five weeks.

If you missed any of the "Sunday Night Football" action, The Sporting News had you covered with live updates, highlights and more throughout the game.

Vikings vs. Broncos final score

1 2 3 4 F Vikings 0 10 7 3 20 Broncos 3 6 0 12 21

Vikings vs. Broncos results, highlights from 'Sunday Night Football'

(All times Eastern)

Final: Broncos 21, Vikings 20

11:25 p.m.: TURNOVER ON DOWNS — Joshua Dobbs stepped into a big throw across the middle but no one was home. That'll do it. The Broncos will come away with a
comeback victory.

11:24 p.m.
</item 18>

<item 19>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 19>

<item 20>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
He's already missed from 48 tonight.

End of third quarter: Bears 6, Vikings 3

10:14 p.m. — The Bears are starting to look respectable on third down again, this time completing to D.J. Moore to move deeper into Vikings territory. Chicago
still might need another first down to give Cairo Santos a fair shot at a field goal.

10:08 p.m. INTERCEPTION — Dobbs throws his third interception of the night! Jaylon Johnson deflected it, Edwards caught it, and interestingly, the Bears have
less favorable field position than they would've if it had fallen incomplete.

And Edwards follows it up with the Bears 3rd INT of the game ‼️
</item 20>

<item 21>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
MORE: Inside the Cowboys historic offensive explosion vs. Giants

11. Seattle Seahawks 6-3 (13)

The Seahawks needed every ounce of their offense around Geno Smith to rally to beat the Commanders, but the get-well performance worked well for him, Kenneth
Walker and the wide receivers. The defense could be more consistent.

12. Minnesota Vikings 6-4 (14)

The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running
game for Kevin O'Connell and the defense overall under Brian Flores. Their 0-3 and 1-4 starts are distant memories as the Vikings look for a sixth straight win
in Week 11 vs. Denver.

13. Houston Texans 5-4 (16)

The Texans are looking great with C.J. Stroud leading their offense and now they've found some running game, too.
</item 21>

<item 22>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
COURTLAND SUTTON INCREDIBLE 4TH DOWN GRAB ‼️



🎥: @NFL

pic.twitter.com/T0Mrch9Tf9 — The Sporting News (@sportingnews) November 20, 2023

Vikings 20, Broncos 15

11:02 p.m.: FIELD GOAL — Greg Joseph knocks it in from 30 yards. The Vikings' offense was moving, and it did its job by chewing a lot of clock on a 14-play,
seven-minute and 20-second drive. Minnesota stalled out on Denver's 12-yard line, settling for a field goal.

Can the Vikings' defense keep the Broncos out of the end zone as it has all night? The drive of the game is coming up.

10:57 p.m.: The spot on the field was overturned! Ty Chandler was ruled just short of the line to gain, setting up a 4th & inches.
</item 22>

<item 23>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:04 p.m. — Dobbs is picked off again! That's two interceptions and just nine passing yards for the Vikings quarterback, but this one doesn't look like his
fault - Jordan Addison saw it bounce off his hands. It's Jaquan Brisker with the pick.

9:01 p.m. — Another penalty gifts the Vikings a first down... Both teams are piling on the mistakes.

8:58 p.m. — A roughing-the-passer call goes against the Bears and gives the Vikings some breathing room early in the drive.

8:54 p.m. — Fields overthrows Darnell Mooney on third down, and Mooney gets destroyed in a collision. The Vikings will get the ball back, still somehow only
trailing by three. Fields will want that one back.

8:50 p.m. INTERCEPTION — Jaylon Johnson intercepts a throw from Dobbs! That just hung up there.
</item 23>

<item 24>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
9:21 p.m.: Joshua Dobbs and center Garrett Bradbury have had a few bad exchanges when the newly acquired QB is under center. Dobbs just fumbled his second snap
of the game but has recovered both of them.

Vikings 10, Broncos 6

9:18 p.m.: FIELD GOAL — Will Lutz gets another 31-yarder to go for the Broncos. Russell Wilson found Jerry Jeudy for a five-yard gain on 3rd & 7, coming just
short of the line to gain. Head coach Sean Payton elected to kick the field goal instead of going for it on Minnesota's 12-yard line.

9:14 p.m.: Russell Wilson and Courtland Sutton connect on a 33-yard pitch and catch to give the Broncos their first big play of the game! Denver is in the red
zone for the first time tonight.

Vikings 10, Broncos 3

9:08 p.m.
</item 24>

<item 25>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 25>

<item 26>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
No one has scored a touchdown yet tonight.

10:40 p.m. — Addison's catch stands, and an Alexander Mattison run has the Vikings at the edge of field goal range. They want more than three points, but it's
early enough that they absolutely could settle for a field goal if it got to that point.

10:37 p.m. — Matt Eberflus throws the challenge flag on a nice grab by Addison to give the Vikings a first down. This one looks like it will be tough to
overturn...

10:34 p.m. — Wow, Jordan Addison would've had a walk-in touchdown, but Dobbs' throw takes him out of bounds. Addison couldn't have been more wide open.

10:33 p.m. — Brandon Powell has been the Vikings' biggest playmaker at two key points tonight. He goes 18 yards here to put Minnesota near midfield.

10:29 p.m.
</item 26>

<item 27>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:30 p.m. — The Bears' defense has suddenly morphed back into an elite unit on these last two plays, forcing intentional grounding by Dobbs and pressuring him
heavily on the following play. It's 3rd & 24 for the Vikings, who might just settle for a gain of a few yards to set up the field goal.

9:27 p.m. — Another first down by K.J. Osborn drives the Vikings into Bears territory, and the Vikings suddenly find themselves 13 yards from the end zone after
Kyler Gordon is called for interfering with T.J. Hockenson. The Vikings have a chance to take the lead with 38 seconds on the clock.

9:25 p.m. — What an effort by Brandon Powell to turn around and make a 28-yard grab while falling to the ground. That's the biggest offensive play of the night
for the Vikings.

A perfect catch from Brandon Powell on a perfect pass
</item 27>

<item 28>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Denver takes over from its own 34 on a much-needed turnover.

10:13 p.m.: The Broncos offense is really struggling to move the ball on some of these drives tonight. It's another three-and-out after Russell Wilson couldn't
link up with Marvin Mims Jr. on 3rd & 3. The Vikings will take over from their own 25 with 6:23 remaining in the quarter after a touchback on the punt.

Vikings 17, Broncos 9

10:06 p.m.: TOUCHDOWN! Joshua Dobbs scrambles for 10 yards into the end zone to give the Vikings a 17-9 lead.

10:04 p.m.: Broncos CB Patrick Surtain II was flagged for a blatant defensive pass interference call on an interception on 3rd & 13. Automatic first down for
Minnesota, bringing it into the red zone.

10:01 p.m.: The Vikings are moving the chains.
</item 28>

<item 29>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 29>

<item 30>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Broncos 3, Vikings 0

8:30 p.m.: FIELD GOAL — Will Lutz buries a 31-yarder to get the Broncos on the board. The Vikings' defense held strong after the turnover. LB Danielle Hunter
came up with a timely sack to force a 3rd & 15 and the Broncos couldn't convert.

8:26 p.m.: FUMBLE! Vikings QB Joshua Dobbs attempted to pick up a first down on 3rd & 1 and he was absolutely popped by SS Kareem Jackson. LB Baron Browning
pried the ball loose as Dobbs was hit and the Broncos recovered the fumble, taking over at Minnesota's 30.

8:24 p.m.: The Vikings received to start the game and after a touchback on the opening kick, we are underway!

Pregame

8:21 p.m.: Broncos legend Peyton Manning shows some love to Vikings star WR Justin Jefferson ahead of the game.
</item 30>
=== END ITEM id=e3dc0402149e7ffb ===
