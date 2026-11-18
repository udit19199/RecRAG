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

This file has 20 items.

=== ITEM id=a047-6c83344732b43510 ===
Gold sentences and facts:
Sentence 0: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.1: Epic's focus on painting Google as the bad guy happened on day one of the trial.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 2>

<item 3>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I don’t know if that was on behalf of you and fellow developers. Can you tell us what that email was actually about?

Google didn’t leak the email; they leaked their ridiculously biased summary of it to Abner Li at 9to5Google. Don Harrison testified at trial that he didn’t
think Google leaked to the press. And then he was presented with a document including Google’s press team and Sameer Samat, head of all Android, basically
summarizing the articles that appeared as a result of Google’s leaking of our plans to the press.

So I was wondering if you could tell me what the email actually said.

Oh, it’s in evidence! This was the one I mentioned! Right before Christmas-ish 2019, we were planning to launch the new Marvel season with a bunch of Star Wars
content and its awesome lineup. I asked Google’s executives to let Fortnite come to the Google Play Store using our own payment method and to open up, you know,
using their own payment methods to all developers, not just Epic.
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
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
As an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one developer that wanted to go the
route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020, worth approximately $360 million,
to keep the developer on Google Play, Epic will argue.

Google, however, will characterize this program as a way to incentivize developers to launch their apps on Android at the same time as they launch on iOS and
plans to dispute Epic’s claims that it prevented developers from opening their own app stores.

Android allows sideloading

Google will make note that, unlike Apple, it also allows apps to be sideloaded on Android devices — something Epic takes advantage of today. It will note this
process is streamlined and helps keep users safe.
</item 5>

<item 6>
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
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 8>

<item 9>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 10>
=== END ITEM id=a047-6c83344732b43510 ===

=== ITEM id=a128-fdaeff2237b6a965 ===
Gold sentences and facts:
Sentence 0: Heart of Darkness (1899) is a novella by Polish-British novelist Joseph Conrad, about a voyage up the Congo River into the Congo Free State, in the heart of Africa, by the story's narrator Charles Marlow.
  Fact 2.3: Heart of Darkness is about a voyage up the Congo River into the Congo Free State.

Retrieved items:
<item 1>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 1>

<item 2>
Season of Migration to the North
Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist
Tayeb Salih.

 Originally published in Arabic in 1966, it has since been translated into more than twenty languages.

 Salih was fluent in both English and Arabic, but chose to pen this novel in Arabic.

 The English translation was published in 1969 as part of the influential Heinemann African Writers Series.

 The novel is a counter narrative to "Heart of Darkness".

 It was described by Edward Said as one of the six great novels in Arabic literature.

 In 2001 it was selected by a panel of Arab writers and critics as the most important Arab novel of the twentieth century.
</item 2>

<item 3>
First North Americans
First North Americans are a series of historical fiction novels published by Tor and written by husband and wife co-authors W. Michael Gear & Kathleen O'Neal
Gear.

 The series, which began with 1990's "People of the Wolf", explores various civilizations and cultures in prehistoric North America.

 It is comparable to Jean M. Auel's "Earth's Children" series, set in prehistoric Europe, but each of its books focuses on a different time period, location,
and set of characters.

 The first four novels form a coherent, more or less linear narrative, from the initial migration of Siberian peoples into what is now Canada and Alaska (People
of the Wolf) through the florescence of the Mississippian semi-urban mound-building culture, considered the "high-water mark" of North American pre-Columbian
civilization, around 1000 AD.

 The remaining novels cover a wide variety of times and settings, in no particular order, ranging from tropical Florida in the 6th millennium BC to the Chaco
Empire of the Southwest in the 13th century AD.
</item 3>

<item 4>
Graph query row:
work: Heart of Darkness
relationship: CREATED_BY
connected_entity: Joseph Conrad
</item 4>

<item 5>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 5>

<item 6>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 6>

<item 7>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 7>

<item 8>
First North Americans
The novels take into account new developments in North American archaeology such as the discovery of Kennewick Man and the development of the coastal-route
model as a possible alternative or supplement to overland migration across Beringia.
</item 8>

<item 9>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 9>

<item 10>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 10>
=== END ITEM id=a128-fdaeff2237b6a965 ===

=== ITEM id=a081-aa6f511794a55af0 ===
Gold sentences and facts:
Sentence 0:  U2 did not play the song live until 2 July 2009, when they performed it on the second concert of their U2 360° Tour in Barcelona.
  Fact 0.1: U2 performed the song on 2 July 2009 on the second concert of their U2 360° Tour in Barcelona.

Retrieved items:
<item 1>
Psycho Circus World Tour
Psycho Circus World Tour was a Kiss concert tour in 1998–2000.

 It was the first concert tour in history to have 3-D visual effects.

 The Smashing Pumpkins opened at the Dodger Stadium show only, in costume as The Beatles for the Halloween night performance.

 The Dodger Stadium show was streamed live on the internet as well as a radio broadcast.

 Two songs, "Psycho Circus" and "Shout It Out Loud", were screened live on Fox television as part of the "Kiss Live: The Ultimate Halloween Party" special.

 The vast majority of songs in the setlist were played on the previous Alive/Worldwide concert dates, leading to some frustration from fans expecting the return
of classic songs not played on the previous tour.

 Peter Criss was quoted in Metal Edge magazine at the time as wanting to add "Parasite" to the setlist.

 The tour was initially hyped as having circus-style acts as pre-show entertainment.

 This ultimately happened only at the first concert at Dodger Stadium.
</item 1>

<item 2>
Rock &amp; Roll Over Tour
Rock and Roll Over Tour is a concert tour by the American hard rock group Kiss.

 It began November 24, 1976 (shortly after the release of the Rock and Roll Over album) and ended April 4, 1977.

 It was on this tour where the band performed for the first time at Madison Square Garden on 2/18/77 - a venue all four members of the band had long dreamed of
playing (February 18, 1977).

 The band also performed in Japan for the first time on this tour.

 They played Budokan Hall four nights in a row breaking the previous concert attendance record set by The Beatles.

 On December 12, 1976 Ace Frehley suffered a severe electrical shock on stage which delayed the concert.

 The delay lasted about fifteen minutes and then resumed; Ace was able to finish the show despite having lost feeling in one of his hands.

 Being inspired by the event, Ace wrote the song "Shock Me" which appeared on Kiss' next album release Love Gun (1977).

 It is the first Kiss song in which Ace sings lead vocals.
</item 2>

<item 3>
U2 360° Tour
The U2 360° Tour was a worldwide concert tour by rock band U2.

 Launched in support of the group's 2009 album "No Line on the Horizon", the tour visited stadiums from 2009 through 2011.

 The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.

 To accommodate the stage configuration, a large four-legged structure nicknamed "The Claw" was built above the stage, with the sound system and a cylindrical,
expanding video screen on top of it.

 At 164 ft tall, it was the largest stage ever constructed.

 U2 claimed that the tour would be "the first time a band has toured in stadiums with such a unique and original structure."
</item 3>

<item 4>
Psycho Circus World Tour
They used a translator on stage to let the crowd understand exactly what Stanley was saying.

 At the end of the performance, the band ignited all of the pyrotechnics at once; as a result, they were banned from performing in Bremen.

 Ticket sales for this tour were notably slower than the previous Reunion Tour, with many of the smaller market shows underselling and a second North American
leg for the summer of 1999 cancelled all together, the band ultimatly decided on embarking on a Farewell tour in the new millennium.
</item 4>

<item 5>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 5>

<item 6>
World on Fire World Tour
The World on Fire World Tour is the third concert tour by lead guitarist Slash as a solo artist, which started in July 2014 and was scheduled to resume in late
2015, in support of Slash's third solo album "World on Fire".

 The tour features the same backing band that performed with Slash during his two last tours, billed as "Slash featuring Myles Kennedy and The Conspirators",
featuring Myles Kennedy handling lead vocal, bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 6>

<item 7>
Escape Tour
The Escape Tour was a concert tour by the American rock band Journey.

 It was one of Journey's most successful concert tours, in support of their first number one album, "Escape".

 The tour included six consecutive sold out dates at the Pine Knob Theatre in Detroit, and four straight sold out shows at the Los Angeles Forum, and Chicago's
Rosemont Horizon.

 Journey also made an appearance on July 2, 1982 at the Rose Bowl in Pasadena, California with Blue Öyster Cult, Triumph and Aldo Nova.

 The 8 ⁄ -month tour took Journey through Japan and North America.

 Point Blank, Billy Squier, the Greg Kihn Band and Loverboy were the opening acts during the tour.
</item 7>

<item 8>
The Year of the Voyager
The Year Of The Voyager is a double DVD/CD released by Nevermore.

 It was released in Europe on October 20, 2008 and in North America on November 25, 2008 via Century Media.

 The set covers the "This Godless Endeavor" touring cycle, starting with live footage from the U.S. Gigantour 2005, the Metal Mania festival 2006 in Poland, the
Wacken Open Air festival in 2006 in Germany and the main DVD show recorded at the Zeche in Bochum, Germany.

 Bonus material includes two songs from Century Media USA 10th Anniversary Party 2001, all promo videos and an interview with singer Warrel Dane which was
recorded at the Roax Film Studios in Berlin in the spring of 2008.
</item 8>

<item 9>
Psycho Circus World Tour
This ultimately happened only at the first concert at Dodger Stadium.

 Peter Criss later said that it didn't work out because the circus performers wanted equal billing and that some had even wanted to use KISS' backstage dressing
room.

 "2,000 Man" was played to bring in the new Millennium at the 1999/2000 New Year's Eve show at Vancouver, advertised at the time as being recorded for Alive IV.

 The Vancouver show was also notable as being the first time the original members had played non-original band era material live in concert – "I Love It Loud",
"Lick It Up" and "Heaven's on Fire" were added to the setlist and subsequently played on the Farewell Tour.

 "Forever" was listed on concert setlists at the Vancouver show but was not played.

 It was thought at the time it may have been intended as a Paul Stanley solo version prior to "Black Diamond".

 One notable show on the tour was the March 12 Bremen, Germany, show.

 After the opening song, Paul Stanley announced that the local fire marshall had banned Kiss from using any pyrotechnics during the show.
</item 9>

<item 10>
Because We Can (concert tour)
Because We Can was a concert tour by American rock band, Bon Jovi.

 The tour was in support of the band's twelfth studio album "What About Now".

 The tour was named after the lead single from "What About Now".

 All five major continents were visited during the 2013 tour.

 The tour saw the band travel to Wales for the first time since their 2001 One Wild Night Tour and was also the first time since their 1995 These Days Tour that
the band returned to Africa for two dates in South Africa.

 Furthermore, this tour was the first time in the band's 30-year history that they visited Bulgaria and Poland.

 The tour ranked 1st on Pollstar's annual "Top 100 Mid Year Worldwide Tours".

 It earned $142.1 million from 60 shows.

 At the end of 2013, the tour placed 1st on Pollstar's "Top 100 Worldwide Tours", grossing $259.5 million from 102 shows.
</item 10>
=== END ITEM id=a081-aa6f511794a55af0 ===

=== ITEM id=a090-cdb914ed738d48f4 ===
Gold sentences and facts:
Sentence 0: John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.
  Fact 1.0: John Alan Lasseter was born January 12, 1957.

Retrieved items:
<item 1>
Wreck-It Ralph
Wreck-It Ralph is a 2012 American 3D computer-animated fantasy-comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 52nd Disney animated feature film.

 The film was directed by Rich Moore, who has directed episodes of "The Simpsons" and "Futurama", and the screenplay was written by Phil Johnston and Jennifer
Lee from a story by Moore, Johnston, and Jim Reardon.

 John Lasseter served as the executive producer.

 The film features the voices of John C. Reilly, Sarah Silverman, Jack McBrayer, and Jane Lynch.

 The film tells the story of the eponymous arcade game villain who rebels against his role and dreams of becoming a hero.

 He travels between games in the arcade and ultimately must eliminate a dire threat that could affect the entire arcade and one that Ralph himself
unintentionally started.
</item 1>

<item 2>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 2>

<item 3>
John Lasseter (born 1957) is an American animator, director, writer, and producer who led creative work at Pixar, Disney Animation, and DisneyToon, and advised
Walt Disney Imagineering. He directed Pixar’s pioneering shorts *Luxo Jr.* and *Red’s Dream*, and made his feature debut with 1995’s *Toy Story*. He
co-developed the *Toy Story* story and directed *Cars 2*; other listed films include *Toy Story
</item 3>

<item 4>
Lasseter Family Winery
Lasseter Family Winery is a winery located in Glen Ellen, Sonoma County, California.

 The winery was founded in 2000 by Pixar and Walt Disney Animation Studios CCO John Lasseter and his wife, Nancy Lasseter.

 The winery, once inhabited by the Grand Cru Winery, produces approximately 1,200 cases of French red wine blends annually, with the capacity to produce up to
6,000.

 The winery grows Bordeaux and Rhône varietals on 27 acres.

 One of the Lasseters' winemaking mentors was Jess Jackson, of Kendall-Jackson.
</item 4>

<item 5>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 5>

<item 6>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 6>

<item 7>
Red's Dream
Red's Dream is a 1987 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The short film, which runs four minutes, stars Red, a unicycle.

 Propped up in the corner of a bicycle store on a rainy night, Red dreams about a better place.

 "Red's Dream" was Pixar's second computer-animated short following "Luxo Jr." in 1986, also directed by Lasseter.
</item 7>

<item 8>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 8>

<item 9>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 9>

<item 10>
Walt Disney Pictures, founded under that name in 1983, produces live-action films; Walt Disney Studios Motion Pictures distributes Disney studio releases,
succeeding Buena Vista (1953–2007). Disney highlights include *Bambi* (1942), *Pinocchio* (1940), and the 1989–99 Renaissance, spanning ten animated films from
*The Little Mermaid* to *Tarzan*. Buddy Baker scored several Disney films. Other entries cover Mickey and Donald, the
</item 10>
=== END ITEM id=a090-cdb914ed738d48f4 ===

=== ITEM id=a102-403c213f13ffd184 ===
Gold sentences and facts:
Sentence 0: Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
  Fact 1.0: Cyrtanthus is a genus of perennial plants.

Retrieved items:
<item 1>
Ammocharis
Ammocharis is a small genus from sub-Saharan Africa, in the Amaryllidaceae family (subfamily Amaryllidoideae) which includes 7 species distributed in Africa.

 The plant grows as a succulent, above-ground bulb, preferring seasonally wet, hot, sandy soils and full sun.
</item 1>

<item 2>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 2>

<item 3>
Nerine masoniorum
Nerine masoniorum is a species of flowering plant in the family Amaryllidaceae, subfamily Amaryllidoideae, native to the eastern Cape Province of South Africa.

 It is a bulbous perennial.

 "N. masoniorum" belongs to the group of nerines that have narrow evergreen foliage.

 The thread-like leaves reach a length of 25 cm or more.

 The flowering stem is 15–25 cm tall, with up to 11 flowers arranged in an umbel.

 Each flower has six narrow pink tepals with wavy edges.

 It flowers in late summer in cultivation, the first of the nerines to do so.
</item 3>

<item 4>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 4>

<item 5>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 5>

<item 6>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 6>

<item 7>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 7>

<item 8>
Lycoris radiata
Lycoris radiata, known as red spider lily, red magic lily, or equinox flower, is a plant in the amaryllis family, Amaryllidaceae, subfamily Amaryllidoideae.

 Originally from China, Korea and Nepal, it was introduced into Japan and from there to the United States and elsewhere.

 It is considered naturalized in Seychelles and in the Ryukyu Islands.

 It flowers in the late summer or autumn, often in response to heavy rainfall.

 The common name hurricane lily refers to this characteristic, as do other common names, such as resurrection lily; these may be used for the genus as a whole.
</item 8>

<item 9>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 9>

<item 10>
Amarylloidinae
Amarylloidinae is a now obsolete informal name for an "infrafamily" within the Amaryllidaceae (Amaryllis) family, erected by Hamilton Traub.

 This grouping was designed to fill a perceived gap between the formal rank of subfamily and tribes.

 In his treatment of this family, he divided it first into four subfamiles.

 Within subfamily Amarylloideae he then divided his sixteen tribes into two infrafamilies, Amarylloidinae (12 tribes) and Pancratioidinae (4 tribes), both of
which were subsequently demonstrated to be polyphyletic, and hence were abandoned by Dahlgren, who used no rank between family and tribe.

 On the other hand, he also used a much more restricted Amaryllidaceae corresponding to Traub's subfamily Amarylloideae.

 Thus Traub's Amarylloideae most closely resembles subfamily Amaryllidoideae "sensu" APGIII.
</item 10>
=== END ITEM id=a102-403c213f13ffd184 ===

=== ITEM id=a013-0ae1507504a2b012 ===
Gold sentences and facts:
Sentence 0: On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential ground invasion.
  Fact 0.4: The warning was given in advance of a potential ground invasion.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
BHARAT is not immune to #terrorism and internal enemies @HMOIndia @PMOIndia @AmitShah — Dr. Jaswant Gandhi 🇮🇳 (@JaswantDr) October 7, 2023

However, some speculated that Mossad knew what was going to happen but let Hamas make a mistake so that Israel could launch a ferocious attack on it. "Mossad
was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, but they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful," a user said.

#Mossad was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful
</item 2>

<item 3>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
He said the IDF lost an army base, a general was captured right out of his house. "Invincible "Merkava MK4 MBT's captured, and destroyed...what is wrong with
the Israeli security and defence forces? And (what) MOSSAD did?"

Dr Jaswant Gandhi, a professor of management, said that attack on Israel was a "great Mossad failure". He said India had to learn a lot and keep all
intelligence agencies duly greased and internecine conflict-proof. "Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence
Agencies duly greased and internecine conflict-proof. BHARAT is not immune to #terrorism and internal enemies," he wrote on X.

#Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence Agencies duly greased and internecine conflict proof.
</item 3>

<item 4>
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
</item 4>

<item 5>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
The second objective, at the same time, is to exact an… pic.twitter.com/MzKs7tfv4M — Benjamin Netanyahu - בנימין נתניהו (@netanyahu) October 7, 2023

Reacting to this, however, a user said that knowing the proficiency of the Mossad he found the idea of a "surprise attack" to be a willing lie.

Well, you got the war you wanted Beni...at what cost of Israeli lives you don't care. Knowing the proficiency of the Mossad I find the idea of a "surprise
attack"to be a willing lie on your side of this. https://t.co/35qRvgBJt2 — timm (@timmainsson) October 7, 2023

Ben Muasya, a journalist, said that Mossad is overrated.
</item 5>

<item 6>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"How come they never knew Hamas militants planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate heavily? We wait to see how it goes
but it will be messy."

Mossad is overated how comes they never knew this Hamas militants and had planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate
heavily?We wait to see how it goes but it will be messy. pic.twitter.com/W5vpoylus2 — Ben Muasya. (@muasya_ben) October 7, 2023

Another user said that no one can carry out such a major attack against Israel without intelligence support. "Who gave Hamas Intelligence? It was as if the
entire Israeli army and the Mossad were asleep."
</item 6>

<item 7>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
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
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
On Saturday, October 7, for example, suspicious movement in Gaza convinced the Shin Bet head to send a special team to the southern border, but he was unable to
convince the top IDF echelons to prepare accordingly. To handle those weak signals, Eiland claims, one must use "scenario theory", meaning, taking into account
scenarios that have a low probability of occurring, but with devastating results should they occur, and the simple way to prepare for them.

Eiland explains: "You detect certain noises on Friday night, but conclude that the probability of an attack is low, let's say only 10%. Those who act only on
the probability do not consider the question: if this attack happens - will you be able to live with yourself for eliminating this threat? There’s also a third
consideration, which is ease of effort: what is the effort required of me to prepare for this unlikely but dangerous scenario?
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a013-0ae1507504a2b012 ===

=== ITEM id=a023-e1270e3b27ca7b29 ===
Gold sentences and facts:
Sentence 0: Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana.
  Fact 0.0: The U.S. will travel to GEODIS Park in Nashville, Tennessee on October 17.

Retrieved items:
<item 1>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 1>

<item 2>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 2>

<item 3>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 3>

<item 4>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: Breaking down Germany squad, tactics as Julian Nagelsmann debuts as manager

USA roster for October friendlies vs Germany, Ghana

The USMNT squad for the October friendlies was announced on Thursday, October 5. Gregg Berhalter has called in 23 players, which is one fewer than were brought
in for the friendlies against Uzbekistan and Oman last month.

Below is the official 23-man roster for the upcoming October friendlies against Germany and Ghana.

PSV midfielder Malik Tillman was forced to withdraw from the squad due to injury, with Alex Zendejas of Club America named to the roster in his place.

MORE: Tracking every goal scored by Christian Pulisic at new club AC Milan

Pos Name Club Caps Goals GK Ethan Horvath Nott. Forest (ENG) 9 0 GK Matt Turner Nott. Forest (ENG) 33 0 DEF Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino
Dest PSV Eindhoven
</item 4>

<item 5>
Gregg Berhalter named a 23-player USMNT squad for October friendlies: Germany in Hartford on Oct. 14, Julian Nagelsmann’s first match as coach, and Ghana in
Nashville on Oct. 17. Gio Reyna returned after injury and his 2022 World Cup fallout with Berhalter; Lennard Maloney earned his first senior call-up. Tyler
Adams and Antonee Robinson were unavailable through injury, while Malik Tillman withdrew and Alex Zende
</item 5>

<item 6>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 6>

<item 7>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 7>

<item 8>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 9>

<item 10>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Gio Reyna looked sharp in the starting lineup for the United States men’s national team, in all the ways a young man can: his unyielding energy, his surprising
eagerness to cover the whole of the midfield, the periodic flashes of skill when the opportunity presented itself, even the bleach-blond hair that allowed
viewers to spot him in an instant.

There was one substantial element missing from his performance, though, in an immense friendly Saturday against Germany: being able to endure from starting to
finishing.

It was kind of like that for the USMNT as a whole, as they blew an early lead and fell, 3-1, in East Hartford, Conn.

MORE: USMNT undone by errors in loss to Germany

It is quite possible this was the most challenging friendly the U.S.
</item 10>
=== END ITEM id=a023-e1270e3b27ca7b29 ===

=== ITEM id=a087-4b37e0035ad5471b ===
Gold sentences and facts:
Sentence 0: Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian mathematician.
  Fact 1.3: Vladimir Igorevich Arnold died on 3 June 2010.

Retrieved items:
<item 1>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 1>

<item 2>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 2>

<item 3>
Cantor–Bernstein theorem
In set theory and order theory, the Cantor–Bernstein theorem states that the cardinality of the second type class, the class of countable order types, equals
the cardinality of the continuum.

 It was used by Felix Hausdorff and named by him after Georg Cantor and Felix Bernstein.

 Cantor constructed a family of countable order types with the cardinality of the continuum, and in his 1901 inaugural dissertation Bernstein proved that such a
family can have no higher cardinality.
</item 3>

<item 4>
Georg Cantor
Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.

 He invented set theory, which has become a fundamental theory in mathematics.

 Cantor established the importance of one-to-one correspondence between the members of two sets, defined infinite and well-ordered sets, and proved that the
real numbers are more numerous than the natural numbers.

 In fact, Cantor's method of proof of this theorem implies the existence of an "infinity of infinities".

 He defined the cardinal and ordinal numbers and their arithmetic.

 Cantor's work is of great philosophical interest, a fact of which he was well aware.
</item 4>

<item 5>
Alfred von Arneth
Alfred Ritter von Arneth (10 July 181930 July 1897) was an Austrian historian.

 His principal scholarly work is a ten-volume biography of the Habsburg Empress Maria Theresa, first published in installments from 1863 to 1879 and still
regarded as the standard work on the subject.

 Born at Vienna, he was the son of Joseph Calasanza von Arneth (1791–1863), a well-known historian and archaeologist, who wrote a history of the Austrian Empire
(Vienna, 1827) and several works on numismatics and brother of Doctor Franz Hektor von Arneth (1818–1907).
</item 5>

<item 6>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 6>

<item 7>
Cantor space
In mathematics, a Cantor space, named for Georg Cantor, is a topological abstraction of the classical Cantor set: a topological space is a Cantor space if it is
homeomorphic to the Cantor set.

 In set theory, the topological space 2 is called "the" Cantor space.

 Note that, commonly, 2 is referred to simply as the Cantor set, while the term Cantor space is reserved for the more general construction of "D" for a finite
set "D" and a set "S" which might be finite, countable or possibly uncountable.
</item 7>

<item 8>
The entries cover Ravenel’s stable-homotopy conjectures (largely proved except the open telescope conjecture); linguistic binomial pairs; mathematician Hermann
Brunn and Brunnian links; Jónsson–Tarski algebras; and Cantor-related topics: the Cantor–Bernstein theorem, proper classes, Cantor space, the symbol Tav for an
inconsistent collection of cardinals, Cantor’s mathematics and first set-theory paper, the fat Cantor
</item 8>

<item 9>
Jónsson–Tarski algebra
In mathematics, a Jónsson–Tarski algebra or Cantor algebra is an algebraic structure encoding a bijection from an infinite set "X" onto the product "X"×"X".

 They were introduced by .

 , named them after Georg Cantor because of Cantor's pairing function and Cantor's theorem that an infinite set "X" has the same number of elements as "X"×"X";
the term "Cantor algebra" is also occasionally used to mean the Boolean algebra of all clopen subsets of the Cantor set, or the Boolean algebra of Borel subsets
of the reals modulo meager sets (sometimes called the Cohen algebra).
</item 9>

<item 10>
Georg Cantor's first set theory article
Georg Cantor's first set theory article was published in 1874 and contains the first theorems of transfinite set theory, which studies infinite sets and their
properties.

 One of these theorems is "Cantor's revolutionary discovery" that the set of all real numbers is uncountably, rather than countably, infinite.

 This theorem is proved using Cantor's first uncountability proof, which differs from the more familiar proof using his diagonal argument.

 The title of the article, "On a Property of the Collection of All Real Algebraic Numbers" ("Ueber eine Eigenschaft des Inbegriffes aller reellen algebraischen
Zahlen "), refers to its first theorem: the set of real algebraic numbers is countable.
</item 10>
=== END ITEM id=a087-4b37e0035ad5471b ===

=== ITEM id=a097-de77bfe23b2f35df ===
Gold sentences and facts:
Sentence 0: Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist, and literary critic.
  Fact 2.3: Michel Foucault was a French philosopher.

Retrieved items:
<item 1>
Dramaturgy (sociology)
Dramaturgy is a sociological perspective commonly used in microsociological accounts of social interaction in everyday life.

 The term was first adapted into sociology from the theatre by Erving Goffman, who developed most of the related terminology and ideas in his 1959 book, "The
Presentation of Self in Everyday Life".

 Kenneth Burke, whom Goffman would later acknowledge as an influence, had earlier presented his notions of dramatism in 1945, which in turn derives from
Shakespeare.

 However, the fundamental difference between Burke's and Goffman's view is that Burke believed that life was in fact theatre, whereas Goffman viewed theatre as
a metaphor.

 If we imagine ourselves as directors observing what goes on in the theatre of everyday life, we are doing what Goffman called dramaturgical analysis, the study
of social interaction in terms of theatrical performance.
</item 1>

<item 2>
Ray Birdwhistell
Collaborations with others, including initially Margaret Mead and Gregory Bateson, and later, Erving Goffman and Dell Hymes had huge influence on his work.

 For example, the book he is best known for, "Kinesics and Context", "would not have appeared if it had not been envisaged by Erving Goffman" and he explicitly
stated "the paramount and sustaining influence upon my work has been that of anthropological linguistics", a tradition most directly represented at the
University of Pennsylvania by Hymes.
</item 2>

<item 3>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 3>

<item 4>
Erving Goffman
Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist
of the twentieth century".

 In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens,
Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
</item 4>

<item 5>
Identity Performance
In everyday interactions, the body serves as a critical site of identity performance.

 In conveying who we are to other people, we use our bodies to project information about ourselves.

 This is done through movement, clothes, speech, and facial expressions.

 What we put forward is our best effort at what we want to say about who we are.

 Yet while we intend to convey one impression, our performance is not always interpreted as we might expect.

 Through learning to make sense of others’ responses to our behavior, we can assess how well we have conveyed what we intended.

 We can then alter our performance accordingly.

 This process of performance, interpretation, and adjustment is what Erving Goffman calls impression management.

 Impression management is a part of a larger process where people seek to define a situation through their behavior.

 People seek to define social situations by using contextual cues from the environment around them.

 Social norms emerge from situational definitions, as people learn to read cues from the environment and the people present to understand what is appropriate
behavior.
</item 5>

<item 6>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 6>

<item 7>
Goffman (disambiguation)
Erving Goffman (1922–1982) was a Canadian sociologist
</item 7>

<item 8>
The Presentation of Self in Everyday Life
The Presentation of Self in Everyday Life is a 1956 sociology book by Erving Goffman, in which Goffman uses the imagery of the theatre in order to portray the
importance of human social interaction.

 Originally published in Scotland in 1956 and in the United States in 1959, it is Goffman’s first and most famous book, for which he received the American
Sociological Association’s MacIver award in 1961.
</item 8>

<item 9>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 9>

<item 10>
The entries survey literature, theatre, film, music, and curiosities: French Decadence favored artifice and excess; Camus staged Dostoyevsky’s *The Possessed*
in 1959; Algren won the 1950 National Book Award; Poe satirized Gothic fiction; Moorcock wrote Elric and science-fiction novels; Nin’s diaries and erotica
appeared partly posthumously. Wilde’s *Dorian Gray* faced censorship and condemnation. Verne’s
</item 10>
=== END ITEM id=a097-de77bfe23b2f35df ===

=== ITEM id=a039-29501e97520624e7 ===
Gold sentences and facts:
Sentence 0: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 1.0: The theory described here is based on the board's language.

Retrieved items:
<item 1>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
When contacted via email, OpenAI’s press relations declined to comment beyond the official blog post. But Microsoft PR gave this statement from Frank Shaw, its
chief communications officer: “We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of
AI to our customers.”

Nadella later published a statement:

“As you saw at Microsoft Ignite this week, we’re continuing to rapidly innovate for this era of AI, with over 100 announcements across the full tech stack from
AI systems, models and tools in Azure, to Copilot. Most importantly, we’re committed to delivering all of this to our customers while building for the future.
We have a long-term agreement with OpenAI with full access to everything we need to deliver on our innovation agenda and an exciting product roadmap; and remain
committed to our partnership, and to Mira and the team. Together, we will continue to deliver the meaningful benefits of this technology to the world.
</item 1>

<item 2>
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
</item 2>

<item 3>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Brockman was invited to join — but it’s unclear whether he’ll take execs up on that invitation.

Board negotiations hit a snag

Bloomberg reports that Lightcap and Murati, among others, are pushing the board to reinstate Altman. But unsurprisingly, the directors are resisting. As of
midday Sunday, the board hadn’t resigned out of concern over who could replace them, and were vetting candidates. One possible new addition could be Salesforce
co-CEO Bret Taylor.

Altman out, Shear in

Altman won’t be returning as CEO, according to a report in The Information citing an internal memo sent by Sutskever. As the search for a new permanent CEO
continues, OpenAI has appointed Emmett Shear, the co-founder of video streaming site Twitch, as interim CEO — replacing Murati.
</item 3>

<item 4>
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
</item 4>

<item 5>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
We are collaborating to figure out the details. Thank you so much for your patience through this. — OpenAI (@OpenAI) November 22, 2023

Altman also posted about the new deal, giving some insight into the roller coaster that we’ve all been riding since his firing was revealed Friday. In his
words, his decision to join Microsoft on Sunday “was the best path for me and the team.” Since then, the new board’s composition and Microsoft’s support appear
to have been enough to bring him back to the AI company he co-founded.

Altman and board in talks

OpenAI’s board of directors is reportedly in talks with Sam Altman, ex-Y Combinator president and an OpenAI co-founder, to return to OpenAI as CEO as soon as
this week.
</item 5>

<item 6>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
OpenAI says that it’ll begin conducting a formal search for a permanent CEO immediately.

“OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence benefits all humanity,” the board wrote in a joint
statement. “The board remains fully committed to serving this mission. We are grateful for Sam’s many contributions to the founding and growth of OpenAI. At the
same time, we believe new leadership is necessary as we move forward. As the leader of the company’s research, product, and safety functions, Mira is
exceptionally qualified to step into the role of interim CEO. We have the utmost confidence in her ability to lead OpenAI during this transition period.”

Sam Altman is a hero of mine. He built a company from nothing to $90 Billion in value, and changed our collective world forever. I can't wait to see what he
does next. I, and billions of people, will benefit from his future work- it's going to be simply incredible.
</item 6>

<item 7>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
</item 7>

<item 8>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a039-29501e97520624e7 ===

=== ITEM id=a103-ef2a6ee1256867b8 ===
Gold sentences and facts:
Sentence 0: Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.
  Fact 1.1: Anatoly Vasilyevich Lunacharsky was born on 23 November 1875 (Old Style 11 November 1875).

Retrieved items:
<item 1>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 1>

<item 2>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 2>

<item 3>
You Fell Victim to a Fateful Struggle
You Fell Victim (Russian: "Вы жертвою пали"; "Vy zhertvoiu pali"), or You Fell Victim to a Fateful Struggle, is a Russian Marxist and revolutionary funeral
march.

 It acted as the funeral dirge of the Russian revolutionary movement, among them the Bolsheviks.
</item 3>

<item 4>
Vpered
Vpered ("Forward" or "Hasten") (1909 - 1912) was an organization emanating from within the Russian Social Democratic Labour Party (Russian Social Democracy or
RSDLP).

 The faction was gathered by Alexander Bogdanov (1873 - 1928) in December 1909.

 The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.

 Although Vpered emerged from the Bolshevik wing of Russian Social Democracy, the group was critical of Lenin.
</item 4>

<item 5>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 5>

<item 6>
Sergey Yustinovich Bagotsky
Sergey Yustinovich Bagotsky (Russian: Серге́й Юстинович Багоцкий ; 15 February 1879 - 15 March 1953) was a Russian Marxist revolutionary and Soviet medicine and
Red Cross diplomat, Soviet head of the Red Cross representative mission to Geneva from 1918 to 1936.
</item 6>

<item 7>
Apollinariya Yakubova
Apollinariya Yakubova (died 1913 or 1917) was a Russian Marxist revolutionary and, with Vladimir Lenin, one of the founders of the League of Struggle for the
Emancipation of the Working Class.
</item 7>

<item 8>
Nikolay Speshnev
Nikolay Alexandrovich Speshnev (Russian: Николай Александрович Спе́шнев; 1821, Kursk - 1882, St. Petersburg) was a 19th-century Russian aristocrat and political
activist, best known for his involvement with the pro-socialist literary discussion group the Petrashevsky Circle.

 He formed a secret revolutionary society from among the members of the circle, which included the young Fyodor Dostoevsky.

 After the government of Tsar Nicholas I arrested the members of the Petrashevsky Circle in 1849, Speshnev was interrogated, threatened with torture, and
eventually sentenced, along with Dostoevsky, Petrashevsky and others, to execution by firing squad.

 The sentence was commuted to hard labour in Siberia, but the prisoners were only informed of this after enduring a mock execution.
</item 8>

<item 9>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 9>

<item 10>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 10>
=== END ITEM id=a103-ef2a6ee1256867b8 ===

=== ITEM id=a109-8a87cd303087b0b2 ===
Gold sentences and facts:
Sentence 0:  The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
  Fact 1.0: The name Sennenhund refers to people called "Senn".

Retrieved items:
<item 1>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 1>

<item 2>
Entlebucher Mountain Dog
The Entlebucher Sennenhund or Entlebucher Mountain Dog is a medium-sized herding dog, it is the smallest of the four Sennenhunds, a dog type that includes four
regional breeds.

 The name Sennenhund refers to people called "Senn", herders in the Swiss Alps.

 Entlebuch is a region in the canton of Lucerne in Switzerland.

 The breed is also known in English as the Entelbuch Mountain Dog, Entelbucher Cattle Dog, and similar combinations.
</item 2>

<item 3>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 3>

<item 4>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 4>

<item 5>
Bernese Mountain Dog
The Bernese Mountain Dog (German: "Berner Sennenhund" ) is a large-sized breed of dog, one of the four breeds of Sennenhund-type dogs from the Swiss Alps.

 The name "Sennenhund" is derived from the German "Senne" ("alpine pasture") and "Hund" ("dog"), as they accompanied the alpine herders and dairymen called
"Senn".

 "Berner" (or "Bernese" in English) refers to the area of the breed’s origin, in the canton of Bern.

 This mountain dog was originally kept as a general farm dog.

 Large Sennenhunde in the past were also used as draft animals, pulling carts.

 The breed was officially established in 1907.

 In 1937, the American Kennel Club recognized it; today, the club classifies it as a member of the Working Group.
</item 5>

<item 6>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 6>

<item 7>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 7>

<item 8>
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 8>

<item 9>
Appenzeller (chicken)
The Appenzeller is a breed of chicken originating in Appenzell region of Switzerland.

 The Appenzeller comes in two varieties.

 The Spitzhauben variety, meaning "pointed hood", has a V-comb and feather crests in males and females.

 The word 'spitzhauben' derives from the frilly hat worn by the women in the Appenzeller region in Switzerland.

 The breed was imported into America by a doctor.

 The Barthuhner ("bearded hen") has a rose comb and no crest.

 Both types appear in either black, golden spangled and silver spangled plumage.

 They are mostly a show breed, but are decent egg layers.

 They lay small white eggs about 5 times per week.
</item 9>

<item 10>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 10>
=== END ITEM id=a109-8a87cd303087b0b2 ===

=== ITEM id=a069-49c9f6a788b34fd7 ===
Gold sentences and facts:
Sentence 0:  Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.
  Fact 0.1: Qionghai refers to a large land area in Hainan.

Retrieved items:
<item 1>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 1>

<item 2>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 2>

<item 3>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 3>

<item 4>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 4>

<item 5>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 5>

<item 6>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 6>

<item 7>
Qionghai
Qionghai () is one of the seven county-level cities of Hainan province, China.

 Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.

 Within this area is the main city, Qionghai City.

 It is located in the east of the island at the mouth of the Wanquan River, 86 km from the provincial capital of Haikou.

 It has an area of 1692 km2 and in 2010, it had a population of 483,217.
</item 7>

<item 8>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 8>

<item 9>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 9>

<item 10>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 10>
=== END ITEM id=a069-49c9f6a788b34fd7 ===

=== ITEM id=a129-a32b7c9ed03b5f76 ===
Gold sentences and facts:
Sentence 0: The Dead Milkmen are an American punk rock band formed in 1983 in Philadelphia, Pennsylvania.
  Fact 1.2: The Dead Milkmen were formed in Philadelphia, Pennsylvania.

Retrieved items:
<item 1>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 1>

<item 2>
Ricky Wilson (American musician)
Ricky Helton Wilson (March 19, 1953 – October 12, 1985) was an American musician best known as the original guitarist and founding member of rock band the
B-52's.

 Born in Athens, Georgia, Wilson was the brother of fellow member Cindy Wilson.

 The B-52's was founded in 1977, when Ricky, his sister Cindy, Kate Pierson, Keith Strickland and Fred Schneider shared a tropical flaming volcano drink at a
Chinese restaurant and, after an impromptu music session at the home of their friend Owen Scott III, played for the first time at a Valentine's Day party for
friends.

.

 Wilson's unusual guitar tunings were a large contribution to the band's quirky sound .
</item 2>

<item 3>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 3>

<item 4>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 4>

<item 5>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 5>

<item 6>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 6>

<item 7>
Dick Delicious and The Tasty Testicles
Dick Delicious and The Tasty Testicles are a comedy / metal band from Atlanta, Georgia.

 The band formed in 1992 and released 3 albums between 1995 and 2003.

 In 2003 the Dick Delicious received the "Howard Stern Award for Musical Excellence" and broke up shortly afterward.
</item 7>

<item 8>
Graph query row:
album: Not Richard, But Dick
band: The Dead Milkmen
city: Philadelphia
</item 8>

<item 9>
Not Richard, But Dick
Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.

 After being out of print for years, Hollywood Records decided to release the album for digital download as of April 2, 2013.
</item 9>

<item 10>
Electric Guitars
Electric Guitars were an English band formed early in 1980 by Neil Davenport (vocals, lyrics) and Richard Hall (bass, vocals) who were both studying English at
Bristol University.

 The band soon increased to a five-man line-up, with Andy Saunders (guitar, vocals), Matt Salt (drums) and Dick Truscott (keyboards), they also later added two
backing singers: Sara and Wendy Patridge.

 Their first single "Health" / "Continental Shelf" was released on local label Fried Egg Records in 1980.
</item 10>
=== END ITEM id=a129-a32b7c9ed03b5f76 ===

=== ITEM id=a118-d0f537a77dc4da2f ===
Gold sentences and facts:
Sentence 0: Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal Navy surface vessels and submarines.
  Fact 0.4: Rosyth Dockyard formerly undertook refitting of Royal Navy surface vessels

Retrieved items:
<item 1>
Goliath (Rosyth)
Goliath is a crane in Rosyth Dockyard, Scotland, with a lift capacity of 1000 t , the largest in Britain.
</item 1>

<item 2>
Queen Elizabeth-class aircraft carrier
The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.

 The first,  , was named on 4 July 2014, with her ship commissioning planned for 2017, and an initial operating capability expected in 2018.

 The second, HMS "Prince of Wales" , is scheduled to be launched around Summer 2017, followed by commissioning in 2020 and service thereafter.

 On 5 September 2014, at the NATO 2014 Wales summit, the Prime Minister David Cameron announced that the second carrier would be brought into service, ending
years of uncertainty surrounding its future.

 This was confirmed by the November 2015 Government Strategic Defence Review, with both carriers entering service, one being available at any time.
</item 2>

<item 3>
Rosyth Dockyard
Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal
Navy surface vessels and submarines.

 Before its privatisation in the 1990s it was formally the Royal Naval Dockyard Rosyth.

 Its primary role is now as integration site for the Royal Navy's newest aircraft carriers - the "Queen Elizabeth"-class.
</item 3>

<item 4>
Rosyth railway station
Rosyth railway station serves the town of Rosyth in Fife, Scotland.

 The station is managed by Abellio ScotRail and lies on the Fife Circle Line, 23.6 km north of Edinburgh Waverley .

 It was opened in 1917 by the North British Railway (as "Rosyth Halt") to serve the nearby naval dockyard.
</item 4>

<item 5>
DFDS
DFDS is Northern Europe's largest shipping and logistics company.

 The company's name is an abbreviation of Det Forenede Dampskibs-Selskab (literally "The United Steamship Company").

 DFDS was founded in 1866, when C.F. Tietgen merged the three biggest Danish steamship companies of that day.
</item 5>

<item 6>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 6>

<item 7>
HMS Prince of Wales (R09)
HMS "Prince of Wales" is the second "Queen Elizabeth"-class aircraft carrier under construction for the Royal Navy, with plans for active service from 2020.

 She is the seventh Royal Navy ship to have the name HMS "Prince of Wales".

 Construction of the ship began in 2011 at Rosyth Dockyard and in April 2016, was said to be around 80% structurally complete.
</item 7>

<item 8>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 8>

<item 9>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 9>

<item 10>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 10>
=== END ITEM id=a118-d0f537a77dc4da2f ===

=== ITEM id=a037-fc659cd00c655b37 ===
Gold sentences and facts:
Sentence 0: Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it.
  Fact 1.2: The Federal Trade Commission and the 17 state attorneys general filed the antitrust lawsuit against Amazon together.

Retrieved items:
<item 1>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Two members of the billionaire family that controls Rogers Communications Inc. are suing the company, saying they’ve been frozen out of board meetings and
blocked from getting information, reigniting a bitter power struggle within one of Canada’s wealthiest clans.

Melinda Rogers-Hixon says she’s the target of a “personal vendetta” by her older brother, Rogers Chair Edward Rogers, according to a letter contained in the
court filing. The Rogers siblings were close to an agreement that would have resolved their longstanding differences, but Edward Rogers reneged on it, according
to the Sept. 20 missive by Rogers-Hixon, which was sent to Chief Executive Officer Tony Staffieri.

The court petition was filed by Rogers-Hixon and her sister, Martha Rogers, and it argues the company has been “oppressive” to them.

The documents show that the fractures inside the prominent Toronto family have never healed.
</item 1>

<item 2>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers. Soon after, then-CEO Joe Natale was sacked and
replaced by Staffieri, the former chief financial officer. Rogers-Hixon and Martha Rogers had opposed the CEO change.

In early 2022, family members agreed to temporarily shelve their disagreements, according to this week’s court filing, to avoid negative publicity as the
company sought regulators’ approval for the takeover of Shaw Communications Inc. The C$20 billion transaction was the company’s biggest deal ever and among the
largest corporate transactions in Canadian history.

Negotiations Collapse

As part of that truce, Rogers-Hixon and Martha Rogers agreed not to participate in board discussions about the Shaw deal or receive materials on it, because of
an argument about Rogers-Hixon’s personal lawyer, whose firm also does work for Telus Corp., a competitor to Rogers and Shaw.
</item 2>

<item 3>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Big Tech stocks tend to be among the hardest hit by high rates, and they were the heaviest weights on the index. Apple fell 2.3 per cent and Microsoft lost 1.7
per cent. Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it. They accuse the
e-commerce behemoth of using its dominant position to inflate prices on other platforms, overcharge sellers and stifle competition. In China, concerns continued
over heavily indebted real estate developer Evergrande. The property market crisis there is dragging on China’s economic growth and raising worries about
financial instability. France’s CAC 40 fell 0.7 per cent, and Germany’s DAX lost 1 per cent.

Crude oil prices rose, adding to worries about inflation. A barrel of benchmark US crude climbed 71¢ to $US90.39. Brent crude, the international standard, added
67¢ to $US93.96 per barrel.
</item 3>

<item 4>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Below is a list of product(s) in your catalog that are not currently eligible to be the Featured Offer because they are not priced competitively compared to
prices for those products from retailers outside Amazon,” according to the email, which was viewed by CNBC. Unbeknownst to him, Chewy was running a discount
promotion, and dropped the price of his product by a few dollars to $39.99 – less than the $43 offer on Amazon. The algorithm had flagged it as a lower offer,
even though the item on Chewy cost $48.54 after shipping and taxes. Ruckel had to make a choice: Lower the price on Amazon or ask Chewy to raise the price of
his product. He opted for the latter.

Fred Ruckel's company Snuggly Cat makes Ripple Rug, an interactive play mat for cats.
</item 4>

<item 5>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.
</item 5>

<item 6>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.
</item 6>

<item 7>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The documents show that the fractures inside the prominent Toronto family have never healed. Rogers Communications is the country’s largest wireless provider
and the owner of a cable television empire, the Toronto Blue Jays baseball team and other media and sports assets. The family’s controlling stake in the public
company is worth close to C$8 billion ($5.9 billion), according to calculations based on company filings for its 2023 annual meeting.

Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto. They’ve dropped 16% this year.

The children of late founder Ted Rogers have frequently fought one another — sometimes driving down Rogers’ stock price as investors grew worried about the
company’s governance.

The bad blood spilled into full view in October 2021, when Edward Rogers publicly battled his sisters and his mother for control of the board and won in court.
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers.
</item 7>

<item 8>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The Shaw takeover finally closed on April 3 of this year, and the Rogers siblings engaged in talks “intended to resolve all issues between us,” Rogers-Hixon
stated in an affidavit. “I was hopeful all issues could be resolved without negative publicity.” But those negotiations collapsed last month, she said.

It isn’t clear from the court filings whether the discussions included the possibility that any Rogers family members would sell their stake in Rogers
Communications, leave their roles at the company, or exit the family trust that controls almost all of the voting shares of the public company.

“This matter should be resolved privately,” Rogers Communications spokesperson Sarah Schmidt said by email. “We’ve demonstrated that we will not be distracted
by these actions — we have significant momentum in the market, our merger is tracking ahead of plan, and we remain squarely focused on doing what’s right for
our customers and stakeholders.”
</item 8>

<item 9>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.
</item 9>

<item 10>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Amazon in recent years made the forum anonymous, but users must have a seller account in order to post. Another post included a screenshot of a message Amazon
sent to sellers the day after the FTC filed its complaint, which said, “As your partners, we know that this news may generate questions for you and our business
together. This lawsuit does not change anything about our relationship with you or how we operate today.” One user called it “BS verbiage,” adding, “Businesses
that sell in their store are indeed customers. And which of us has gotten good customer service?” Another user described their experience in the last 12 months
of selling on Amazon as “being up all night at an effing casino but I'm stuck, the drugs are starting to wear off, but I'm trying to break even on the mortgage
payment I'm using to play. That's how it is selling on Amazon right now to me.” The seller went on to describe the experience as a “race to the bottom.”
</item 10>
=== END ITEM id=a037-fc659cd00c655b37 ===

=== ITEM id=a017-53db8553aaa4dc86 ===
Gold sentences and facts:
Sentence 0: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 0.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 5>

<item 6>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 7>

<item 8>
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
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 10>
=== END ITEM id=a017-53db8553aaa4dc86 ===

=== ITEM id=a071-1ec3a90468813680 ===
Gold sentences and facts:
Sentence 0: The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I Football Bowl Subdivision (FBS) (formerly I-A).
  Fact 1.0: The Mountain West Conference (MW) is a collegiate athletic conference.

Retrieved items:
<item 1>
Men's National Collegiate Athletic Association
The Men's National Collegiate Athletic Association is an athletic association in the Philippines, and is the direct counterpart of the older Women's National
Collegiate Athletic Association (founded in 1970).

 The Men's National Collegiate Athletic Association (MNCAA) is exclusive for men.

 The MNCAA was founded in 2004.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 The competing sports of the tournament are Basketball, Volleyball, Futsal, Table Tennis, Badminton and Streetdance.
</item 1>

<item 2>
2011–12 Wagner Seahawks men's basketball team
The 2011–12 Wagner Seahawks men's basketball team represented Wagner College during the 2011–12 NCAA Division I men's basketball season.

 The Seahawks, led by second year head coach Dan Hurley, played their home games at Spiro Sports Center and are members of the Northeast Conference.

 They finished the season 25–6, 15–3 in NEC play to finish in second place.

 They lost in the semifinals of the NEC Basketball Tournament to Robert Morris.

 Despite having 25 wins, the Seahawks did not accept an invitation to a post season tournament.
</item 2>

<item 3>
2013–14 Iona Gaels men's basketball team
The 2013–14 Iona Gaels men's basketball team represented Iona College during the 2013–14 NCAA Division I men's basketball season.

 The Gaels, led by fourth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 22–11, 17–3 to win the MAAC regular season championship.

 They advanced to the championship game of the MAAC Tournament where they lost to Manhattan.

 As a regular season conference champion who failed to win their conference tournament, they received an automatic bid to the National Invitation Tournament
where they lost in the first round to Louisiana Tech.
</item 3>

<item 4>
2012 Hawaii Warriors football team
The 2012 Hawaii Warriors football team represented the University of Hawaii at Manoa in the 2012 NCAA Division I FBS football season.

 They were led by first-year head coach Norm Chow and played their home games at Aloha Stadium.

 They were first year members of the Mountain West Conference.

 They finished the season 3–9, 1–7 in Mountain West play to finish in a tie for ninth place.
</item 4>

<item 5>
Hawaii Rainbow Warriors baseball
The Hawaii Rainbow Warriors baseball team is a varsity intercollegiate athletic team of the University of Hawaii at Manoa in Honolulu, Hawaii, United States.

 The team is a member of the Big West Conference, which is part of the National Collegiate Athletic Association's Division I.

 Hawaii's first baseball team was fielded in 1923.

 The team plays its home games at Les Murakami Stadium in Honolulu, Hawaii.

 The Rainbow Warriors are coached by Mike Trapasso.
</item 5>

<item 6>
List of Pacific Tigers head football coaches
The Pacific Tigers college football team represented University of the Pacific (CA).

 The Tigers competed in the National Collegiate Athletic Association (NCAA) College Division in the years 1937–1968.

 In 1969, the team moved to National Collegiate Athletic Association (NCAA) Division I and stayed there until disbanding after the 1995 season.
</item 6>

<item 7>
Wayne State Warriors football
The Wayne State Warriors football team is the college football team at Wayne State University in Detroit, Michigan.

 The Wayne State football team played their first game in October 1918.

 The Wayne State Warriors have competed in the Great Lakes Intercollegiate Athletic Conference since 1999 (and previously from 1975-1989), and are currently a
Division II member of the National Collegiate Athletic Association (NCAA).

 Wayne State plays their home games at Tom Adams Field at Wayne State Stadium.

 All Wayne State games are broadcast on WDTK radio.
</item 7>

<item 8>
Women's National Collegiate Athletic Association
The Women's National Collegiate Athletic Association (WNCAA) is an athletic association in the Philippines exclusively for women.

 It was founded in 1970.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 Its men's counterpart is the Men's National Collegiate Athletic Association, founded in 2004.
</item 8>

<item 9>
Mountain West Conference
The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I
Football Bowl Subdivision (FBS) (formerly I-A).

 The MW officially began operations in July 1999.

 Geographically, the MW covers a broad expanse of the Western United States, with member schools located in California, Colorado, Hawaii, Idaho, Nevada, New
Mexico, Utah, and Wyoming.

 Craig Thompson has served as Commissioner of the MW since its founding in 1999.
</item 9>

<item 10>
2012–13 Wagner Seahawks men's basketball team
The 2012–13 Wagner Seahawks men's basketball team represented Wagner College during the 2012–13 NCAA Division I men's basketball season.

 The Seahawks were led by the youngest men's head coach in NCAA Division I, Bashir Mason, who was 28 when he was elevated from an assistant position in March
2012 following the departure of Dan Hurley for Rhode Island.

 The Seahawks played their home games at Spiro Sports Center and were members of the Northeast Conference.

 They finished the season 19–12, 12–6 in NEC play to finish in a three way tie for second place.

 They advanced to the semifinals of the Northeast Tournament where they lost to Long Island.

 For the second consecutive year, despite a winning record, Wagner choose not to participate in a post season tournament.
</item 10>
=== END ITEM id=a071-1ec3a90468813680 ===

=== ITEM id=a005-01a4cdd6887b3c41 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.2: Google makes deals with Samsung.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
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
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 4>

<item 5>
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
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 7>

<item 8>
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
</item 8>

<item 9>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a005-01a4cdd6887b3c41 ===

=== ITEM id=a108-f585a6ff68a099b5 ===
Gold sentences and facts:
Sentence 0: The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.
  Fact 0.3: The TD Garden is in Boston, Massachusetts.

Retrieved items:
<item 1>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 1>

<item 2>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 2>

<item 3>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 3>

<item 4>
Alutiiq Museum
The Alutiiq Museum or Alutiiq Museum and Archaeological Repository is a non-profit museum and cultural center dedicated to preserving and sharing the cultural
traditions of the Koniag Alutiiq branch of Sugpiaq ~ Alutiiq of the Alaska Native people.

 The museum is a state-of-the-art facility and is located on the first floor of the "Alutiiq Center" of Kodiak, Alaska.

 Alutiiq Museum is one of three museums in Kodiak.

 The museum is the seventh museum in Alaska and the second "tribal museum" in the United States to be accredited.

 The museum provides tours of its exhibits, laboratory and collections storage facilities to educational groups.

 The museum will accept materials relevant to the prehistoric, historic, and contemporary cultural history of the Native peoples who settled the Koniag Alutiiq
Nation.

 Such materials include, but are not limited to, archaeological, ethnological, photographic, film, audio, archival, and natural history specimens.
</item 4>

<item 5>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 5>

<item 6>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 6>

<item 7>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 7>

<item 8>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 8>

<item 9>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 9>

<item 10>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 10>
=== END ITEM id=a108-f585a6ff68a099b5 ===
