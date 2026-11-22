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

=== ITEM id=3179564aa37b561b ===
Gold sentences and facts:
Sentence 0: The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.
  Fact 0.0: The agreement also meant that Apple never switched to a Google competitor.
  Fact 0.1: The agreement also meant that Apple never allowed users to choose their browser when setting up their iPhones.
Sentence 1: Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months.
  Fact 1.0: Apple has already indicated that it plans to appeal.
  Fact 1.1: Appealing is a lengthy process.
  Fact 1.2: Appealing can take around 18 months.
Sentence 2: They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.
  Fact 2.0: The party described uses its own store.
  Fact 2.1: The party described uses its own payments.
  Fact 2.2: The party described forces developers to all have the same terms.
  Fact 2.3: The party described forces OEMs to all have the same terms.
  Fact 2.4: The party described forces carriers to all have the same terms.

Retrieved items:
<item 1>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
In 2013, then President Barack Obama vetoed an import ban on the iPhone. At the time, Samsung accused Apple of infringing on its cellular data patents. But in
that case, according to Levi, Apple was able to make the argument that there were issues regarding fair, reasonable, and nondiscriminatory licensing (FRAND).
The patents involved were considered standard and essential, and the Obama administration was concerned the import ban would give Samsung “undue leverage.”
Plus, it only impacted the iPhone 4 and older models of the iPad. Newer iPhones didn’t use an infringing chipset. In the case of the Apple Watch, Apple doesn’t
have any of these arguments.

“In that case 10 years ago, Apple really rolled the dice and the pulled the rabbit out of the hat. It’s very unlikely for them to do that again,” says Levi.

Where does Apple go from here?
</item 1>

<item 2>
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
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 5>

<item 6>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
And after the 24th, they’ll disappear from Apple Stores, too.

“I think Apple sees the writing on the wall and they’re preparing themselves,” says Ben Levi, a partner at Levi Snotherly & Schaumberg, which has experience
litigating ITC disputes.

It’s rare to see Big Tech lose — and with such tangible consequences at that. There are still ways that Apple can keep the watches on sale, but it’ll likely
take a lot of waiting or a lot of money to make it happen.

How Apple got here

This ban is the result of a long-standing patent dispute between Apple and the medical device maker Masimo. The latter is known for its pulse oximetry tech,
generally referred to as SpO2 or blood oxygen saturation in the wearable world, and it claims that Apple is using that patented technology without permission.

This particular story started about 10 years ago when Apple reached out to Masimo about a potential partnership around blood oxygen features on its wearables.
</item 6>

<item 7>
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
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 8>

<item 9>
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
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 10>
=== END ITEM id=3179564aa37b561b ===

=== ITEM id=9363ac89cb757ebd ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made these minor physical upgrades alongside the big improvements.
  Fact 0.2: Valve's new iteration includes big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units of the Steam Deck OLED ready to ship on that same day.

Retrieved items:
<item 1>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 1>

<item 2>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 2>

<item 3>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 4>

<item 5>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 5>

<item 6>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 7>

<item 8>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 8>

<item 9>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 9>

<item 10>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 10>
=== END ITEM id=9363ac89cb757ebd ===

=== ITEM id=94b81f2f7f372710 ===
Gold sentences and facts:
Sentence 0: You're probably starting him if Kenneth Walker III remains out with his oblique injury, but consider other options if you have the depth.
  Fact 0.0: Kenneth Walker III has an oblique injury.
  Fact 0.1: Kenneth Walker III may remain out with his oblique injury.
  Fact 0.2: You are probably starting him if Kenneth Walker III remains out with his oblique injury.
  Fact 0.3: Consider other options if you have the depth.
Sentence 1: With Kenneth Walker III's oblique injury comes opportunity for the rookie, who already looked strong in the month leading up to Seattle's narrow Week 11 loss to the Rams.
  Fact 1.0: Kenneth Walker III has an oblique injury.
  Fact 1.1: Kenneth Walker III's oblique injury creates opportunity for the rookie.
  Fact 1.2: The rookie already looked strong in the month leading up to Seattle's narrow Week 11 loss to the Rams.
  Fact 1.3: Seattle lost narrowly to the Rams in Week 11.
Sentence 2: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 2.0: Kenneth Walker III plays for the Seahawks.
  Fact 2.1: Kenneth Walker III took four carries against the Rams.
  Fact 2.2: Kenneth Walker III left the game with an oblique injury.
  Fact 2.3: Kenneth Walker III left the game after taking four carries against the Rams.

Retrieved items:
<item 1>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Here's looking at who to start in seasonal leagues and DFS lineups from the latest edition of TNF:

MORE THURSDAY NIGHT FOOTBALL:

DK lineup | FD lineup | ATS pick | Best props

Seahawks: Who to start, sit in fantasy football on TNF

Sit: QB Geno Smith

Smith has dropped from the top 12 last season to cumulative QB20 in 2023. He's had two great starts and three so-so ones, but the rest has been forgettable. The
Cowboys have allowed only two strong fantasy performances by quarterbacks, (Brock Purdy and Jalen Hurts), and we don't expect Smith to make it three.

Sit: RB Zach Charbonnet

There was a black fly in the Charbonnet against the 49ers last week.
</item 1>

<item 2>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
Keep rolling with Alexander Mattison (vs. Bears), too, against another below-average run defense.

Welcome to the top 25, Zach Charbonnet (vs. 49ers)! With Kenneth Walker III's oblique injury comes opportunity for the rookie, who already looked strong in the
month leading up to Seattle's narrow Week 11 loss to the Rams. Sure, Charbs' 69-yard effort in Los Angeles last week was a bit underwhelming, but we love his
upside with a full game as the Seahawks' bell-cow.

Speaking of bell-cow backs, things are looking very promising for the short-term workload of Green Bay's AJ Dillon (@ Lions). Veteran Aaron Jones (knee) and
rookie Emanuel Wilson (shoulder) both got hurt in Week 11, so Dillon should be the main man in Motown for the Pack.
</item 2>

<item 3>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Tampa Bay has a pretty good run defense, so Allgeier can't be trusted outside of a desperate flex play with his nine average touches over the past two weeks.

What are the biggest RB injuries in Week 14?

Derrick Henry (head/neck), Rhamondre Stevenson (ankle), Kenneth Walker III (oblique), Aaron Jones (knee), D'Onta Foreman (ankle), and D'Andre Swift
(undisclosed) are all "questionable," at best, to suit up this week. We know for sure Stevenson will be out, but we'll have to monitor the injury reports
throughout the week for the rest.

Jonathan Taylor (thumb) underwent surgery last week and will miss the next few weeks, opening the door for Zack Moss to resume his role as Indy's super-sub lead
back.

We'll be updating these RB PPR rankings all week, so check back for the latest player movement.
</item 3>

<item 4>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Kansas City's front-seven has been tough against speedsters and more vulnerable to power backs, and the 5-11, 190-pound Cook 100 percent falls into the former
category. Temper your expectations.

We feel the same way about Zach Charbonnet (@ 49ers). It's great that the Week 13 injury he sustained on Thursday night was just a bruise, but it's not great
that he now faces a 49ers defense that already stifled him on Thanksgiving. You're probably starting him if Kenneth Walker III remains out with his oblique
injury, but consider other options if you have the depth. The floor and ceiling remain low against this uber-confident Niners D.

Tyler Allgeier (vs. Buccaneers) continues to see his role diminish for the NFC South-leading Falcons, with dum-dum coach Arthur Smith finally realizing that
Bijan Robinson gives the ATL the best chance at continued success.
</item 4>

<item 5>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 5>

<item 6>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.
</item 6>

<item 7>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
This is the good advice that you should take: Be wary of anyone who's rushed for just 47 yards two straight weeks despite getting significant duty with Kenneth
Walker III (oblique) on the shelf. Charbonnet's also not doing much in the passing game, and this is as bad of a matchup as San Francisco was last week. You
should do better, even in a big bye.

WEEK 13 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Start: WR DK Metcalf

Metcalf can flash big plays and stay involved in games...until he disappears when Smith can't get the ball to him. Smith might be judicious about throwing his
way with second-year corner sensation DaRon Bland in coverage. That said, you simply can't sit DeKaylin Zacharius against anyone because of his big-game
ability.
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 8>

<item 9>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?
</item 9>

<item 10>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 10>
=== END ITEM id=94b81f2f7f372710 ===

=== ITEM id=a8133846714d988c ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Caroline Ellison is Bankman-Fried's on-again off-again girlfriend.
  Fact 0.1: Caroline Ellison is the CEO of Alameda.
  Fact 0.2: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.3: According to the government, Bankman-Fried had secret access to customer money, both cash and crypto.
Sentence 1: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 1.0: The second week of the trial's standout testimony came from Caroline Ellison.
  Fact 1.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 1.3: Caroline Ellison claimed the $14 billion was used to repay debts to lenders.
  Fact 1.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 1.5: Caroline Ellison claimed she did this under the instruction of SBF.

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
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 2>

<item 3>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Much of said funding came from customers, and Easton testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make
investments or direct funding toward political causes and charities.

SBF trial: What we learned in week two

The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to
repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also went into detail about how a $2 billion venture fund, FTX
Ventures, was funded by money that had previously been allocated to Alameda from third-party lenders.

And though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1 billion in
frozen trading accounts.

SBF trial: What we learned in the week one

As expected, the opening statements in the SBF trial were spicy.
</item 3>

<item 4>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 6>

<item 7>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 8>

<item 9>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Then a bigger one. Then an effectively limitless one.

Ellison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have believed you if you told me I would
be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more comfortable with.”

Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When FTX took a loss in
the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s “insurance fund,” meant to
keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.

Even that wasn’t good enough.
</item 10>
=== END ITEM id=a8133846714d988c ===

=== ITEM id=34beb37d7c223aa3 ===
Gold sentences and facts:
Sentence 0: Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest, Sophie.
  Fact 0.0: Zazie Beetz will return in the role of Arthur's neighbor.
  Fact 0.1: Zazie Beetz's character Sophie is Arthur's love interest.
  Fact 0.2: Sophie is unwitting about being Arthur's love interest.
Sentence 1: But then, there’s MJ as well who has made a comeback from the first installment.
  Fact 1.0: MJ is also in the new installment.
  Fact 1.1: MJ has made a comeback from the first installment.

Retrieved items:
<item 1>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Mary’s gameplay missions end up increasing the overall quality of Spider-Man 2. All things considered, it is great to see Insomniac improving on this aspect
instead of removing it completely.

ADVERTISEMENT Article continues below this ad

Marvel’s Spider-Man 2 is now available on PlayStation 5 worldwide.

Watch This Story: The Console Wars Are Heating Head-to-Head: Xbox and PlayStation Fans Battle Over Which Custom Controller is Better, Starfield or Spider-Man 2?
</item 1>

<item 2>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Coming back to the ‘Wake Up’ mission, she’s seen sneaking past the hunters but she’s also capable of performing Takedowns on them. It wasn’t the case in the
previous title. Her missions have become better now with the devs actually spending time on giving her a chance to fight.

Coming back to the stun gun part, MJ is pretty skilled in using it, during missions like ‘Good Men’, thanks to her training with Sable. Also, now she’s got the
option to even run away and hide when spotted by enemies. The playability factor around her has increased a lot compared to being just an observer, for the most
part, in the first game. That’s the biggest and most visible difference.

Mary Jane and her missions in Spider-Man 2 allow the story to proceed without causing a problem for the central protagonists

The best thing about going through the quests showcasing Mary Jane is that they never try to overpower the ongoing story itself.
</item 2>

<item 3>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
There’s a visible growth in her arc especially considering her profession as a journalist. Yes, she’s Peter’s love interest, but that’s not everything she’s
there for. Players can easily see it happening throughout the game’s duration.

The verdict?

ADVERTISEMENT Article continues below this ad

It’s obvious to expect that a lot of fans who didn’t like MJ’s missions in the first installment won’t like her this time as well. But considering how she was
before and how she’s in Spider-Man 2, the overall change is quite praiseworthy. Things are different this time, thankfully.

MJ’s missions are now having some actual meaning attached to them rather than acting like fillers. Players who want to have a few interesting moments aside from
the main quests featuring Miles and Peter can have a sense of finishing well-designed stealth quests. Mary’s gameplay missions end up increasing the overall
quality of Spider-Man 2.
</item 3>

<item 4>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
2024

Joker: Folie à Deux: Oct. 4

Todd Phillips’ acerbic 2019 drama Joker — effectively a DC-themed riff on Martin Scorsese’s King of Comedy — didn’t necessarily call for a sequel. It ends in a
wide-open place that invites viewers to write their own next chapter. But nothing breeds sequels like success, and the first Joker was a billion-dollar hit for
DC during a period where it was struggling to find that kind of payday from any of its properties. It feels like DC may have handed Phillips a blank check and a
free rein — at least, that seems like the best way to explain why the Joker sequel is reportedly a musical thriller with ​​Joaquin Phoenix reprising his role as
Arthur Fleck/the Joker, and Lady Gaga taking on the role of Harley Quinn. Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest,
Sophie.
</item 4>

<item 5>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
MJ in Spider-Man 2 vs. MJ from Insomniac’s 2018 installment

While there are similarities between the character’s iterations in two titles, a lot of drastic differences are also present. Speaking of similar aspects, the
first game had a mission where Mary gets to use a taser to shock the guards. Well there’s no taser in Spider-Man 2, she’s got a stun gun now.

Then, there are moments of hiding and sneaking past the enemies. In the original game, there were missions like the one in Grand Central Station where MJ had to
stay hidden from a number of Mr. Negative’s demons to defuse a bomb. Similarly, in Spider-Man 2, MJ is seen hiding from the hunters during the ‘Wake Up’ quest.
But what’s actually changed?

Well, the very difference lies in the lack of MJ’s abilities in the first game and how she’s evolved in the second.
</item 5>

<item 6>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.
</item 6>

<item 7>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
One can even call her the third unofficial superhero in the latest PS5 exclusive.

A refreshing take on Mary Jane’s stealth missions makes her a strong character in Marvel’s Spider-Man 2

ADVERTISEMENT Article continues below this ad

In the first title, stepping into MJ’s shoes was a great concept, but the devs somehow couldn’t execute the plan flawlessly. As a result, missions featuring her
as a playable character were heavily criticized by most of the fans. There was nothing too special about those quests whatsoever.

Now, looking at Mary Jane in Marvel’s Spider-Man 2, there’s a very clear difference one can make out. In short, it feels comparing night and day. Being a
journalist at The Daily Bugle, her character is bound to go places in order to investigate mysteries. The first game had such situations, but the sequel
improved on them. It gives MJ a lot more strength instead of being someone who always needs to be rescued in the last moment. But how?
</item 7>

<item 8>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
With the recent release of Marvel’s Spider-Man 2, Insomniac has witnessed a huge appreciation from fans for delivering a memorable sequel. Not to mention the
epic portrayal of Peter Parker and Miles Morales fighting against some of the strongest villains. But then, there’s MJ as well who has made a comeback from the
first installment. And it’s her arc that has actually received an impressive elevation this time. Especially when considering the poorly received missions
featuring Mary Jane in the original game from 2018.

America’s Favorite Video Today

So what’s actually changed in 2023 that has created a vast difference? There are certain key reasons making Mary Jane in Spider-Man 2 a treat to watch. Not just
from the perspective of her relationship with Peter, but her own missions showcasing a different side this time. One can even call her the third unofficial
superhero in the latest PS5 exclusive.
</item 8>

<item 9>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
(If you can get past the Frankenstein-ification of the boroughs, that is. RIP Chelsea.)

There are hints of greater thematic aspirations within Spider-Man 2. Questions about mass incarceration, recidivism, and reintegration into society for formerly
incarcerated people; worries about the surveillance state and what we give up when we engage with modern technology. These things appear, however fleetingly, in
Spider-Man 2, but there isn’t enough time to elaborate on any single one in the unrelenting march of set-piece after glorious set-piece.

But I’m glad they’re there, anyway. Because if we’re going to retell a story, we ought to retell it anew. Canons were made to be broken, and Spider-Man 2 swings
in that direction. With a sequel teed up by the game’s final act, I’m eager to see where Insomniac goes with that momentum. Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct.
</item 9>

<item 10>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
But how?

Compared to the lack of interactivity in 2018’s Spider-Man, MJ is now capable of tackling the enemies around her. In the first title, all she could do was to
move from one point to the other. Now, she’s able to do more than that – make pretty good and lethal moves. To be precise, the devs have given MJ a little
firepower, if not in too much capacity. Now, she’s got weapons instead of being clueless in moments of being vulnerable against armed enemies. Yes, she hasn’t
got any superpowers, but she’s got other means to help herself out.

Due to such involvement, players have now got a reason to be more invested into a given mission focusing on MJ. It wasn’t the case at all taking into
consideration her previous iteration in Insomniac’s original game. Mary Jane can now handle the circumstances in her own manner, thus breaking free from the
linear approach of just trying to hide and escape.
</item 10>
=== END ITEM id=34beb37d7c223aa3 ===

=== ITEM id=11c591506af47452 ===
Gold sentences and facts:
Sentence 0: After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry at his family’s response to it.
  Fact 0.0: William learned of Diana’s death from his father.
  Fact 0.1: William learned of Diana’s death at the royal residence of Balmoral in Scotland.
  Fact 0.2: The character William is played by Rufus Kampa.
  Fact 0.3: William is left heartbroken with grief.
  Fact 0.4: William is angry at his family’s response to Diana’s death.
Sentence 1: The latest installment takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
  Fact 1.0: The latest installment takes place between the late 1990s and the mid-2000s.
  Fact 1.1: The latest installment includes the lead-up to Princess Diana’s fatal Paris car crash.
  Fact 1.2: Princess Diana’s fatal Paris car crash happened in 1997.

Retrieved items:
<item 1>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
With the show heading into the late Nineties and early Noughties in its final season, which arrived on Netflix on Thursday (16 November), events are being
depicted on screen that many of the public will remember well.

The first part of season six takes place in the summer of 1997, and sees Prince Charles (Dominic West) building on his relationship with Camilla Parker Bowles
(Olivia Williams), while Princess Diana (Elizabeth Debicki) sparks up a romance with film producer Dodi Fayed (Khalid Abdalla). The show also features the
pair’s death in a car crash in Paris.

As the new season arrives, and fans wonder what in The Crown is based in reality, here’s everything you need to know.

Diana’s role with Tony Blair’s government

Episode one sees Diana visiting Tony (Bertie Carvel) and Cherie Blair (Lydia Leonard) with William after asking for a meeting with him.
</item 1>

<item 2>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
It was later believed to have been delivered to their room before they went to Dodi’s flat, where a ring bearing the words “Dis-moi Oui” (“Tell me Yes”) – also
the name of episode three – was later recovered, alongside a receipt for a “bague de fiançaille” (engagement ring). Speaking at the inquest, his father Mohamed
claimed that the pair had met the jeweller in Monte Carlo during the holiday on the yacht, and that he believed Dodi planned to propose that night.

Prince William goes missing in Balmoral after Diana’s death

After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry
at his family’s response to it. At one point, Charles is informed that the prince is not in his room and that “no one can find him”, prompting Charles and
Prince Harry to look for him and a subsequent hunt across the grounds for the young royal.
</item 2>

<item 3>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.

William, then 15, and Harry, then 12, inquire over the phone when their mother will be coming back home from Paris. “I’m coming home tomorrow, and I’m seeing
you both tomorrow night,” Diana replies in the episode, which also takes place on the same day as her fatal car crash later that evening. The boys then ask
Diana whether she plans on marrying Dodi, considering their relationship has been plastered all over the tabloids.

Diana’s poignant last words to William and Harry shown in The Crown scene

“Well, you know better than to believe the papers,” she tells William, before adding: “I’m emphatically not going to marry Dodi. To be honest, I can’t wait to
come home.”

When William sweetly asks his mother if she’s doing “okay”, Diana candidly tells her eldest son: “I’m okay. It’s just a bit mad here.
</item 3>

<item 4>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
He is not located, but later walks back to the house on his own accord, drenched in rain. “14 hours, that poor boy was gone,” the Queen later says.

William (Kampa) goes missing after his mother’s death (Keith Bernstein)

It’s true that William and Harry initially stayed in Balmoral with their grandmother following the death of Diana. While it’s not known whether he went missing,
during a visit to Scotland in 2021, William recalled how that experience was one of the “saddest” moments of his life. “I was in Balmoral when I was told that
my mother had died. Still in shock, I found sanctuary in the service at Crathie Kirk that very morning,” he recalled. “And in the dark days of grief that
followed, I found comfort and solace in the Scottish outdoors. As a result, the connection I feel to Scotland will forever run deep.”

The Crown season six part one is on Netflix now, with part two arriving on 14 December.
</item 4>

<item 5>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Through tears, she claimed that he had proposed to her (something his family denied) and promised to buy her a house in Malibu. However, she dropped the lawsuit
following Dodi’s death.

Diana’s close relationship with paparazzi on France yacht trip

When Dodi and Diana first spend time together on his father’s yacht in France, the paparazzi are never far away. However, when William refuses to go outside
until the photographers leave them alone, she takes a boat out to visit them. Wearing a swimming costume, she asks them: “We’re having a lovely time, apart from
one little thing: you lot.” “Don’t be like that, you love us really,” they reply. Diana asks them to leave her alone as they’re “freaking out” her sons, before
telling one photographer – who she knows by name – that if they go away, “you’re going to get a big surprise with the next thing I do”.
</item 5>

<item 6>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
“The last call between Harry, William and Diana is heartbreaking,” a third user commented.

In both the series and in real-life, Diana and Dodi - along with their chauffeur, Henri Paul - were killed on 31 August 1997 when their Mercedes-Benz was
pursued by paparazzi and crashed in the Pont de l’Alma tunnel in Paris. The mother of two was 36 years old.

The fatal car crash was recreated in the very first scene of season six. While the incident is not shown on-screen, a car carrying Diana and Dodi is seen
speeding past a man walking his dog, before entering the tunnel. As the vehicle’s horn blares, he calls emergency services on his cell phone.

Australian actor Elizabeth Debicki, who took over the role of Diana from Emma Corrin in season five, admitted it was “difficult to recreate” the scene -
especially with her character being hounded by the paparazzi. “It was heavy and very manic, and incredibly invasive.
</item 6>

<item 7>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

The first four episodes of The Crown season six are officially here, but many viewers believe one scene has depicted the most devastating moment throughout the
entire series.

On 16 November, Netflix dropped the first four episodes of The Crown’s final season, followed by the last six episodes on 14 December. The latest installment
takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
</item 7>

<item 8>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
The season begins with Princess Diana (played by Elizabeth Debicki) whisking her sons - Prince William (played by Rufus Kampa) and Prince Harry (played by Fflyn
Edwards) - away on summer holiday to Saint-Tropez, France. After staying several days on Mohamed Al Fayed’s yacht, the boys return home to London where their
father, the then-Prince of Wales, accompanies them to Balmoral Castle to vacation with the rest of the royal family in Scotland.

In episode three, Diana’s partner Dodi Fayed (played by Khalid Abdalla) surprises her with a quick trip to Paris, France. However, the princess is keen to
return home to the United Kingdom, and is eager to speak to her sons over the phone. A mob of paparazzi following the couple delays Diana’s scheduled phone call
with William and Harry, until she’s able to catch them again later that evening.

It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.
</item 8>

<item 9>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Recalling their discussion to the Queen, Blair says that Diana feels she “still has a lot to offer the country as a public servant”. He says that she had asked
if there was a way to work with the government on “a more formal basis” and that any “official role” would be appreciated.

Bertie Carvel as Blair (Netflix)

Following Diana’s death, Downing Street confirmed that Diana had visited Blair at Chequers with her son a few weeks before, where they discussed a special role
for her as an overseas “ambassador” for Britain. Diana told a journalist at the time that the prime minister had recognised her skills and asked her to
undertake “missions” abroad for Britain.

Queen Elizabeth’s reluctance to accept Camilla

As The Crown season six begins, Charles is frustrated at the public for demonising Camilla long after his split from Diana.
</item 9>

<item 10>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Dodi’s proposal to Diana on the night they died

After bursting into tears going for dinner at the Ritz, move upstairs, he proposes to her with a ring she had previously pointed out when they sheltered from
crowds in a Monte Carlo jewellery shop. She tells him to get up off his knee “I know the whole world is wondering if we’re going to get married, but that’s not
a reason to actually do it.” However, they share a drink and bittersweet conversation about how they need to live their lives going forward - advice

Dodi is shown proposing to Diana before they died (Daniel Escale/Netflix)

In real life, Dodi did not propose to Diana on that fateful evening in Paris. However, at the inquest into the death in 2007, the jury were shown CCTV footage
of him purchasing an engagement ring worth £11,600 in a jewellers across the square from the Ritz on the afternoon of the crash.
</item 10>
=== END ITEM id=11c591506af47452 ===
