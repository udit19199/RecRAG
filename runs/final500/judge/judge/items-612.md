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

This file has 11 items.

=== ITEM id=7af37468327be6a0 ===
Gold sentences and facts:
Sentence 0: Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public school students under the First Amendment to the United States Constitution.
  Fact 0.0: Minersville School District v. Gobitis, 310 U.S. 586 (1940), was a decision by the Supreme Court of the United States.
  Fact 0.1: Minersville School District v. Gobitis involved the religious rights of public school students.
  Fact 0.2: Minersville School District v. Gobitis involved religious rights under the First Amendment to the United States Constitution.
Sentence 1: Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.
  Fact 1.0: Gravel v. United States, 408 U.S. 606 (1972), was a case.
  Fact 1.1: Gravel v. United States concerned the protections offered by the Speech or Debate Clause of the United States Constitution.

Retrieved items:
<item 1>
Forest Grove School District v. T. A.
Forest Grove School District v. T. A., 557 U.S. 230 (2009), is a case in which the United States Supreme Court held that the Individuals with Disabilities
Education Act (IDEA) authorizes reimbursement for private special education services when a public school fails to provide a "free appropriate public education"
(FAPE) and the private school placement is appropriate, regardless of whether the child previously received special education services through the public
school.
</item 1>

<item 2>
Minersville School District v. Gobitis
Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public
school students under the First Amendment to the United States Constitution.

 The Court ruled that public schools could compel students—in this case, Jehovah's Witnesses—to salute the American Flag and recite the Pledge of Allegiance
despite the students' religious objections to these practices.

 This decision led to increased persecution of Witnesses in the United States.

 The Supreme Court overruled this decision a mere three years later, in "West Virginia State Board of Education v. Barnette", 319 U.S. 624 (1943).
</item 2>

<item 3>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 3>

<item 4>
Bethel School District v. Fraser
Bethel School District v. Fraser, 478 U.S. 675 (1986), was a landmark decision by the United States Supreme Court involving free speech in public schools.

 High school student Matthew Fraser was suspended from school in the Bethel School District for making a speech including sexual double entendres at a school
assembly.

 The Supreme Court held that his suspension did not violate the First Amendment.
</item 4>

<item 5>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 5>

<item 6>
Cedar Rapids Community School Dist. v. Garret F.
Cedar Rapids Community School District v. Garret F., 526 U.S. 66 (1999) , was a United States Supreme Court case in which the Court ruled that the related
services provision in the Individuals with Disabilities Education Act (IDEA) required public school districts to fund "continuous, one-on-one nursing care for
disabled children" despite arguments from the school district concerning the costs of the services."

 The judges relied heavily on Irving Independent School Dist.

 v. Tatro, 468 U. S. 883 (1984).

 Under the Court's reading of the IDEA's relevant provisions, medical treatments such as suctioning, ventilator checks, catheterization, and others which can be
administered by non-physician personnel come within the parameters of the special education law's related services.

 Disability advocates considered the Court decision to be a "substantial victory for families of children with disabilities."

 Amendments were made in the Education Flexibility Partnership Act of 1999 to increase IDEA funding as a result of the case.
</item 6>

<item 7>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 7>

<item 8>
Gravel v. United States
Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.

 In the case, the Supreme Court of the United States held that the privileges and immunities of the Constitution's Speech or Debate Clause enjoyed by members of
Congress also extend to Congressional aides, but not to activity outside the legislative process.
</item 8>

<item 9>
Safford Unified School District v. Redding
Safford Unified School District v. Redding, 557 U.S. 364 (2009) , was a United States Supreme Court case in which the Court held that a strip search of a middle
schooler violated the Fourth Amendment to the United States Constitution where the school lacked reasons to suspect either that the drugs (Ibuprofen) presented
a danger or that they were concealed in her underwear.

 The court also held, however, that because this was not clearly established law prior to the court's decision, the officials involved were shielded from
liability by qualified immunity.
</item 9>

<item 10>
Elk Grove Unified School District v. Newdow
Elk Grove Unified School District v. Newdow, 542 U.S. 1 (2004), was a case decided by the U.S. Supreme Court.

 The lawsuit, originally filed as Newdow v. United States Congress, Elk Grove Unified School District, et al. in 2000, led to a 2002 ruling by the United States
Court of Appeals for the Ninth Circuit that the words "under God" in the Pledge of Allegiance are an endorsement of religion and therefore violate the
Establishment Clause of the First Amendment to the United States Constitution.

 After an initial decision striking the congressionally added "one nation under God" language, [Newdow v. U.S. Congress, 292 F.3d 597 (9th Cir.

 2002), the superseding opinion on denial of rehearing en banc was more limited, holding that compelled recitation of the language by school teachers to
students was invalid.

 [328 F.3d 466 (9th Cir.

 2003).
</item 10>
=== END ITEM id=7af37468327be6a0 ===

=== ITEM id=eaf1b9cd550d3bf3 ===
Gold sentences and facts:
Sentence 0:  One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
  Fact 0.0: Updike was one of only three writers to win the Pulitzer Prize for Fiction more than once.
  Fact 0.1: The other two writers who won the Pulitzer Prize for Fiction more than once were Booth Tarkington and William Faulkner.
  Fact 0.2: Updike published more than twenty novels during his career.
  Fact 0.3: Updike published more than a dozen short-story collections during his career.
  Fact 0.4: Updike published poetry during his career.
  Fact 0.5: Updike published art and literary criticism during his career.
  Fact 0.6: Updike published children's books during his career.
Sentence 1:  Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.
  Fact 1.0: Seventeen of Updike's novels were bestsellers.
  Fact 1.1: More than 100 million copies of Updike's books are in print.

Retrieved items:
<item 1>
Tom Clancy's Net Force Explorers: Gameprey
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 1>

<item 2>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 2>

<item 3>
Tom Clancy's Net Force Explorers: Duel Identity
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 3>

<item 4>
Tom Clancy's Net Force Explorers: High Wire
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 4>

<item 5>
Simon Bloom
Simon Bloom is a series of young adult novels by Michael Reisman.

 Set in the fictional town of Lawnville, New Jersey, it is about a boy who finds a book that lets him control the laws of physics.

 The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
</item 5>

<item 6>
Tom Clancy's Net Force Explorers: Deathworld
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 6>

<item 7>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 7>

<item 8>
Tom Clancy's Net Force Explorers: Safe House
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 8>

<item 9>
Tom Clancy's Rainbow 6: Patriots
Tom Clancy's Rainbow 6: Patriots is a cancelled first-person shooter video game, part of the "Tom Clancy's Rainbow Six" series, announced on the cover of the
December 2011 issue of "Game Informer".

 It was to be published by Ubisoft, and was developed by the company's Montreal studio, with additional development by Ubisoft Toronto and Red Storm
Entertainment.

 Due to the death of Tom Clancy in October 2013, concern was raised that this game would become the last to bear his name.

 Ubisoft has since stated that they will continue putting Tom Clancy's name on future Tom Clancy titles out of respect for the late author.
</item 9>

<item 10>
Tom Clancy
Thomas Leo "Tom" Clancy Jr. (April 12, 1947 – October 1, 2013) was an American novelist best known for his technically detailed espionage and military-science
story lines set during and after the Cold War.

 Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.

 His name was also used on movie scripts written by ghost writers, nonfiction books on military subjects, and video games.

 He was a part-owner of the Baltimore Orioles and vice-chairman of their community activities and public affairs committees.
</item 10>
=== END ITEM id=eaf1b9cd550d3bf3 ===

=== ITEM id=8d588f4346cc90b5 ===
Gold sentences and facts:
Sentence 0:  The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
  Fact 0.0: The women's events include the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park.
  Fact 0.1: The women's events include the Roxy Pro at Sunset Beach.
  Fact 0.2: The women's events include the Billabong Pro Maui at Honolua Bay, Maui.
Sentence 1: Haleʻ iwa is a North Shore community and census-designated place (CDP) in the Waialua District of the island of Oʻ ahu, City and County of Honolulu.
  Fact 1.0: Haleʻiwa is a North Shore community.
  Fact 1.1: Haleʻiwa is a census-designated place (CDP).
  Fact 1.2: Haleʻiwa is in the Waialua District of the island of Oʻahu.
  Fact 1.3: Haleʻiwa is in the City and County of Honolulu.

Retrieved items:
<item 1>
1988 Volvo International – Doubles
Jorge Lozano and Todd Witsken won in the final 6–3, 7–6 against Pieter Aldrich and Danie Visser.
</item 1>

<item 2>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 2>

<item 3>
World Match Racing Tour
The World Match Racing Tour is an annual match racing tour and World Championship in the match racing event of sailing.

 The World Match Racing Tour has "special event"-status from the International Sailing Federation.

 The Tour awards over US$1.5 million in prize money, with points awarded at each event culminating in the crowning of the ISAF Match Racing World Champion.

 A US$500,000 overall prize pool for the Championship is now on offer.
</item 3>

<item 4>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 4>

<item 5>
Sport Mundi Tournament
The Sport Mundi Tournament is a Spanish pre-season women's football invitational charitable tournament held every August in Irun since 2005.

 It is contested by four teams, usually including regional powerhouses Athletic Bilbao and Real Sociedad.

 Athletic is the most successful team in the competition with four titles, followed by Levante UD and Real Sociedad with two each.
</item 5>

<item 6>
Ocean City Open
The Ocean City Open is a defunct tennis tournament that was played on the Grand Prix tennis circuit in 1977.

 The event was held in Ocean City, Maryland.

 Vitas Gerulaitis won the singles title while Alex Metreveli and Bill Scanlon partnered to win the doubles title.
</item 6>

<item 7>
Triple Crown of Surfing
The Vans Triple Crown of Surfing is a Hawaiian specialty series of professional surfing events, offering three events to men and three events to women.

 For the men, those events are the Reef Hawaiian Pro at Haleiwa Ali'i Beach Park; the O'Neill World Cup of Surfing at Sunset Beach; and the Billabong Pipeline
Masters at the Banzai Pipeline.

 The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
</item 7>

<item 8>
2012 Monterrey Open
The 2012 Monterrey Open was a women's tennis tournament played on outdoor hard courts.

 It was the 4th edition of the Monterrey Open and was an International tournament on the 2012 WTA Tour.

 It took place at the Sierra Madre Tennis Club in Monterrey, Mexico, from 20 to 26 February.

 Tímea Babos won the singles title.
</item 8>

<item 9>
2001–02 Volvo Ocean Race
The 2001–02 Volvo Ocean Race was the eighth edition of the around-the-world sailing event Volvo Ocean Race, and the first under the name Volvo Ocean Race.

 For the 2001–02 the sponsorship of the race was taken over by Volvo and Volvo Cars.

 The race was renamed the Volvo Ocean Race.

 Stopovers were added in Germany, France, and Sweden being the Volvo's three biggest car markets in Europe.

 In addition the points system had been modified significantly in an effort to keep the race competitive until the final leg.

 The previous "points" race having been effectively won two full legs before the final gun.
</item 9>

<item 10>
Reef Hawaiian Pro
The Reef Hawaiian Pro is a surfing event held annually on the North Shore, Oahu, Hawaii.

 It is currently the first leg of the Triple Crown of Surfing.

 Surfers compete for a share of $135,000 (men) or $35,000 (women) prize money.

 Since it carries a 6-star rating, the event is critical for those who want to qualify for the ASP World Tour.
</item 10>
=== END ITEM id=8d588f4346cc90b5 ===

=== ITEM id=e4fcdace99e7cf2c ===
Gold sentences and facts:
Sentence 0: Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.
  Fact 0.0: Broughtonia is a genus of orchids.
  Fact 0.1: Broughtonia belongs to the family Orchidaceae.
  Fact 0.2: Broughtonia is native to the Bahamas.
  Fact 0.3: Broughtonia is native to the Greater Antilles.
Sentence 1: Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family, Lamiaceae.
  Fact 1.0: Perilla is a genus.
  Fact 1.1: Perilla consists of one major Asiatic crop species, P. frutescens.
  Fact 1.2: Perilla includes a few recognized wild species in nature.
  Fact 1.3: Perilla belongs to the mint family, Lamiaceae.

Retrieved items:
<item 1>
Monardella lanceolata
Monardella lanceolata is a species of flowering plant in the mint family known by the common names mustang mint and mustang monardella.

 It is native to the mountains of California and Baja California, where it grows in chaparral, woodland, rocky slopes, and often disturbed habitat types.
</item 1>

<item 2>
Lamiaceae (mint family) includes aromatic culinary, medicinal, ornamental, seed, and tuber crops. Perilla comprises cross-fertile Asian herbs, notably Korean
perilla and red or green shiso; its perilla ketone can cause fatal pulmonary edema in grazing cattle and horses. Ocimum includes basil and holy basil. Other
described mints include California’s shortspike hedgenettle and mustang mint, and western North America’s medicinal nettleleaf giant hyss
</item 2>

<item 3>
Shiso
Perilla frutescens var.

 "crispa", also called shiso ( , from Japanese シソ ) is a variety of species "Perilla frutescens" of the genus "Perilla", belonging to the mint family,
Lamiaceae.

 The plant occurs in red (purple-leaved) or green-leaved forms.

 Shiso is a perennial plant that may be cultivated as an annual in temperate climates, and occurs in both red- (or purple-) leaved and green forms.

 There are also frilly, ruffled-leaved forms called "chirimen-jiso" and forms that are red only on top, called "katamen-jiso".
</item 3>

<item 4>
Perilla ketone
Perilla ketone is a natural terpenoid that consists of a furan ring with a six-carbon side chain containing a ketone functional group.

 It is a colorless oil that is sensitive to oxygen, becoming colored upon standing.

 The ketone was identified in 1943 by Sebe as the main component of the essential oil of Perilla frutescens.

 Perilla ketone is present in the leaves and seeds of purple mint ("Perilla frutescens"), which is toxic to some animals.

 When cattle and horses consume purple mint when grazing in fields in which it grows, the perilla ketone causes pulmonary edema leading to a condition sometimes
called perilla mint toxicosis.
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Broughtonia
Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.

 The genus is abbreviated Bro in trade journals.
</item 6>

<item 7>
Lamiaceae
or Labiatae is a family of flowering plants commonly known as the mint or deadnettle family.

 Many of the plants are aromatic in all parts and include widely used culinary herbs, such as basil, mint, rosemary, sage, savory, marjoram, oregano, hyssop,
thyme, lavender, and perilla.

 Some species are shrubs, trees (such as teak), or, rarely, vines.

 Many members of the family are widely cultivated, not only for their aromatic qualities but also their ease of cultivation, since they are readily propagated
by stem cuttings.

 Besides those grown for their edible leaves, some are grown for decorative foliage, such as "Coleus".

 Others are grown for seed, such as "Salvia hispanica" (chia), or for their edible tubers, such as "Plectranthus edulis", "Plectranthus esculentus",
"Plectranthus rotundifolius", and "Stachys affinis" (Chinese artichoke).
</item 7>

<item 8>
Perilla frutescens
Perilla frutescens, commonly called perilla or Korean perilla, is a species of "Perilla" in the mint family Lamiaceae.

 It is an annual plant native to Southeast Asia and Indian highlands, and grown in the Korean peninsula, Southern China and India.

 Known as "deulkkae" (들깨 ) in Korean, the plant was introduced into Korea before the Unified Silla era, when it started to be widely cultivated.
</item 8>

<item 9>
Perilla
Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family,
Lamiaceae.

 It encompasses several distinct varieties of Asian herb, seed, and vegetable crop, including "P. frutescens" (deulkkae) and "P. frutescens" var.

 "crispa" (shiso).

 The genus name "Perilla" is also a frequently employed common name ("perilla"), applicable to all varieties.

 Perilla varieties are cross-fertile and intra-specific hybridization occurs naturally.

 Some varieties are considered invasive.
</item 9>

<item 10>
Broussaisia
Broussaisia arguta, the kanawao, is a species of perennial flowering plant in the "Hydrangea" family, Hydrangeaceae, that is endemic to Hawaiʻ i.

 It is the only species in the monotypic genus Broussaisia.
</item 10>
=== END ITEM id=e4fcdace99e7cf2c ===

=== ITEM id=721e6d1b3e254e04 ===
Gold sentences and facts:
Sentence 0:  Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.
  Fact 0.0: Most figures are shaped with the body of Mickey Mouse.
  Fact 0.1: Most figures have different themed markings.
  Fact 0.2: Most figures have different themed colors.
  Fact 0.3: Most figures have different themed patterns.
Sentence 1:  He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.
  Fact 1.0: Mickey Mouse was created by Walt Disney.
  Fact 1.1: Mickey Mouse was created by Ub Iwerks.
  Fact 1.2: Mickey Mouse was created at the Walt Disney Studios.
  Fact 1.3: Mickey Mouse was created in 1928.

Retrieved items:
<item 1>
Hagop Sandaldjian
Hagop Sandaldjian (1931–1990) was an Egyptian-born Armenian American musician and microminiature sculptor, best known for his tiny art pieces displayed at the
Museum of Jurassic Technology in Los Angeles, California.

 Sandaldjian's creations included a carving of Mount Ararat on a grain of rice; a crucifix in which a minute golden figure of Jesus hangs upon a cross made from
a bisected strand of Sandaldjian's own hair; and recreations of Disney figures (Snow White and the Seven Dwarfs or Mickey Mouse, for example) or historical
figures (such as Napoleon or Pope John Paul II) presented in the eye or on the tip of a needle.
</item 1>

<item 2>
Donald Duck
Donald Duck is a cartoon character created in 1934 at Walt Disney Productions.

 Donald is an anthropomorphic white duck with a yellow-orange bill, legs, and feet.

 He typically wears a sailor shirt and cap with a bow tie.

 Donald is most famous for his semi-intelligible speech and his mischievous and temperamental personality.

 Along with his friend Mickey Mouse, Donald is one of the most popular Disney characters and was included in TV Guide's list of the 50 greatest cartoon
characters of all time in 2002.

 He has appeared in more films than any other Disney character, and is the most published comic book character in the world outside of the superhero genre.
</item 2>

<item 3>
Felix the Cat
Felix the Cat is a funny-animal cartoon character created in the silent film era.

 The anthropomorphic black cat with his black body, white eyes, and giant grin, coupled with the surrealism of the situations in which his cartoons place him,
combine to make Felix one of the most recognized cartoon characters in film history.

 Felix was the first character from animation to attain a level of popularity sufficient to draw movie audiences.
</item 3>

<item 4>
Manimal Vinyl
Manimal Vinyl is a Calabasas based record label founded in 2006 by experimental musician and fashion stylist, Paul Beahan.

 They were originally known as a psychedelic folk label that makes semi-annual tribute records for worldwide charities, having eventually switched to dance
music and progressive underground.

 They are the label who has release debut records from act ranging from Warpaint to Bat for Lashes and exclusive singles from Moby, Duran Duran, Carla Bruni,
Edward Sharpe and the Magnetic Zeros.

 In 2008, they founded the annual Manimal Festival in Pioneertown, California which ran from 2008-2010.

 The festival has featured line-ups with Edward Sharpe and the Magnetic Zeros, Warpaint, Ariel Pink and papercranes.

 Manimal has offices in Los Angeles and New York City.
</item 4>

<item 5>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 5>

<item 6>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 6>

<item 7>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 7>

<item 8>
Vinylmation
Vinylmation is a brand of 1.5", 3", and 9" vinyl collectibles sold at Disney theme parks, select Disney Stores, and the online Disney Store.

 The name "Vinylmation" is a combination of the word "Animation" and "Vinyl".

 Most figures are all shaped with the body of Mickey Mouse but have different themed markings, colors, and patterns.

 This means that all figures have round mouse ears, whether or not the painted character has them.

 This is accommodated by painting the ears with backgrounds - stars, stripes, rainbows - to match the painted character.

 The series exception is the Park Starz series which does not resemble Mickey at all, and more closely resembles iconic figures from the parks.

 Vinylmation was first introduced in July 2008, and the first figures were introduced in November 2008.
</item 8>

<item 9>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 9>

<item 10>
Mickey Mouse
Mickey Mouse is a funny animal cartoon character and the official mascot of The Walt Disney Company.

 He was created by Walt Disney and Ub Iwerks at the Walt Disney Studios in 1928.

 An anthropomorphic mouse who typically wears red shorts, large yellow shoes, and white gloves, Mickey has become one of the world's most recognizable
characters.
</item 10>
=== END ITEM id=721e6d1b3e254e04 ===

=== ITEM id=63d9c5a79712ab8c ===
Gold sentences and facts:
Sentence 0:  Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
  Fact 0.0: Dan Hurley is the son of Bob Hurley.
  Fact 0.1: Bob Hurley is a Hall of Fame high school coach.
  Fact 0.2: Dan Hurley is the younger brother of Bobby Hurley.
  Fact 0.3: Bobby Hurley is a former Duke guard.
  Fact 0.4: Bobby Hurley is a former Sacramento Kings guard.
Sentence 1: Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.
  Fact 1.0: Robert Matthew Hurley was born on June 28, 1971.
  Fact 1.1: Robert Matthew Hurley is an American basketball coach.
  Fact 1.2: Robert Matthew Hurley is a former college player.
  Fact 1.3: Robert Matthew Hurley is a former professional player.
Sentence 2:  Hurley is currently the head coach of the Arizona State men's team.
  Fact 2.0: Hurley is currently the head coach of the Arizona State men's team.

Retrieved items:
<item 1>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 1>

<item 2>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 2>

<item 3>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 3>

<item 4>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 4>

<item 5>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 5>

<item 6>
Rhode Island Rams men's basketball
The Rhode Island Rams men's basketball team is a college basketball program that competes in NCAA Division I and the Atlantic 10 Conference representing the
University of Rhode Island.

 The team is under the direction of head coach Dan Hurley, who was hired on March 20, 2012.

 The Rams play their home games at the Ryan Center (capacity 7,657).
</item 6>

<item 7>
Basketball coach David Hobbs served as Alabama head coach (1992–98) and now assists Iowa State. St. John’s Jack Kaiser led baseball to 11 postseason
appearances, helped establish the Big East, and is honored by its tournament award and stadium. Drake’s Dolph Pulliam reached the 1969 Final Four; his No. 5 was
retired. NBA All-Star John Wall was the 2010 top draft pick; high-school standout Zion Williamson drew attention for dunks.
</item 7>

<item 8>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 8>

<item 9>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 9>

<item 10>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 10>
=== END ITEM id=63d9c5a79712ab8c ===

=== ITEM id=3802d113bf3b7bab ===
Gold sentences and facts:
Sentence 0:  In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
  Fact 0.0: In Crash, there is no betting, unlike in Brag.
  Fact 0.1: In Crash, players aim to reach a total of 11 points.
  Fact 0.2: In Crash, the 11 points are gained over successive deals.
Sentence 1: Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.
  Fact 1.0: Three-card brag is a 16th-century British card game.
  Fact 1.1: Three-card brag is the British national representative of the vying or 'bluffing' family of gambling games.

Retrieved items:
<item 1>
Three card brag
Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.

 Brag is a direct descendant of the Elizabethan game of Primero and one of the several ancestors to poker, just varying in betting style and hand rankings.
</item 1>

<item 2>
Ombre
The historical importance of Ombre in the field of playing cards is the fact that it was the first card game in which a trump suit was established by bidding
rather than by the random process of turning the first card of the stock.

 This game developed from Triunfo, though it was from L'Hombre that the idea of bidding was adopted into other card games such as Skat, and Tarot, which owes
Hombre a good portion of its betting system as well.

 The game continued to be in vogue almost in every corner of Europe from the late 17th through the 18th centuries.
</item 2>

<item 3>
Écarté
Écarté is a two-player card game originating from France, the word literally meaning "discarded".

 It is a trick-taking game, similar to whist, but with a special and eponymous discarding phase.

 It is closely related to Euchre, a card game played mainly in the United States.

 Écarté was popular in the 19th century, but is now rarely played.
</item 3>

<item 4>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 4>

<item 5>
Gleek (card game)
Gleek is an English card game for three persons.

 It is played with a 44-card pack and was popular from the 16th century through the 18th century.
</item 5>

<item 6>
Put (card game)
Put is an English tavern trick-taking card game first recorded in the 16th century and later castigated by 17th century moralists as one of ill repute.

 It belongs to a very ancient family of card games and clearly relates to a group known as Trut, "Truque", also "Tru", and the South American game Truco.

 Its more elaborate version is the Spanish game of Truc, which is still much played in many parts of Southern France and Spain.
</item 6>

<item 7>
Quadrille (card game)
Quadrille is a card game that was popular in the 18th century.

 A variant of the Spanish card game Ombre, it is played by four players in pairs, with a deck of 40 cards (the 8's, 9's and 10's being removed).

 By the end of the 19th century, the card game had fallen out of fashion.
</item 7>

<item 8>
Triomphe
Triomphe (French for triumph) is a card game dating from the late 15th century.

 It most likely originated in France or Spain (as triunfo) and later spread to the rest of Europe.

 When the game arrived in Italy, it shared a similar name with the pre-existing game and deck known as "trionfi" (tarot).

 While trionfi has a fifth suit that acts as permanent trumps, triomphe randomly selects one of the existing four suits as trumps.

 Another common feature of this game is the robbing of the stock.

 Triomphe became so popular that during the 16th century the earlier game of trionfi was gradually renamed tarocchi, tarot, or tarock.

 This game is the origin of the English word "trump" and is the ancestor of many trick-taking games like Euchre (via Écarté) and Whist (via Ruff and Honours).
</item 8>

<item 9>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 9>

<item 10>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 10>
=== END ITEM id=3802d113bf3b7bab ===

=== ITEM id=c5dbf1eceb1fbdcc ===
Gold sentences and facts:
Sentence 0: Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
  Fact 0.0: Chalk and Cheese was a short-lived sitcom television show
  Fact 0.1: Chalk and Cheese featured Michael Crawford
  Fact 0.2: Chalk and Cheese featured Robin Hawdon
Sentence 1: Michael Patrick Smith, {'1': ", '2': ", '3': ", '4': "} (born 19 January 1942), known as Michael Crawford, is an English actor, comedian and singer.
  Fact 1.0: Michael Patrick Smith is known as Michael Crawford
  Fact 1.1: Michael Patrick Smith was born 19 January 1942
  Fact 1.2: Michael Crawford is English
  Fact 1.3: Michael Crawford is an actor
  Fact 1.4: Michael Crawford is a comedian
  Fact 1.5: Michael Crawford is a singer

Retrieved items:
<item 1>
A Cream Cracker under the Settee
"A Cream Cracker Under The Settee" is a dramatic monologue written by Alan Bennett in 1987 for television, as part of his "Talking Heads" series for the BBC.

 The series became very popular, moving onto BBC Radio, international theatre, becoming one of the best-selling audio book releases of all time and included as
part of both the A-level and GCSE English syllabus.

 It was the sixth and final episode of the first series of "Talking Heads".
</item 1>

<item 2>
Chalk and Cheese
Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
</item 2>

<item 3>
Matthew Perry
Matthew Langford Perry (born August 19, 1969) is a Canadian-American actor, best known for his role as Chandler Bing on the long-running NBC television sitcom
"Friends", as well as his portrayal of Ron Clark in the 2006 television movie "The Ron Clark Story".
</item 3>

<item 4>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 4>

<item 5>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 5>

<item 6>
The entries profile actors and television productions. Tommy Knight is known for The Sarah Jane Adventures and Doctor Who; Mark Benton played Daniel Chalk in
Waterloo Road; Robin Chalk appeared in Dirty Dancing and Moon. Tamzin Outhwaite starred in EastEnders, Red Cap, Paradox, Vital Signs and Inside No. 9’s “Empty
Orchestra.” Dominic West is known for The Wire and Appropriate Adult; John Altman played Nick Cotton in EastEnders. Other entries cover children’s
</item 6>

<item 7>
John Altman (actor)
John Clarkson Stewart (born 2 March 1952), known as John Altman, is an English actor and singer, perhaps best known for playing "Nasty" Nick Cotton in the
popular BBC soap opera "EastEnders".

 He was among the show's original cast members appearing in the very first episode in February 1985 and appeared on the show on and off as a recurring
character.

 His character was killed off in the 30th anniversary episode of the show which aired in February 2015.

 Altman has also appeared in several other television series and appeared in many stage productions.

 In 2010 he became the new frontman of the band Heavy Metal Kids following the death of former frontman Gary Holton in 1985.
</item 7>

<item 8>
The collection profiles entertainers, creators, and fictional characters across film, television, music, comedy, and magic. Highlights include magician Criss
Angel’s Las Vegas shows; actors such as Michael Crawford, Matthew Perry, Josh Charles, and Glenn Quinn; comedians and creators behind *In Living Color*,
*Upright Citizens Brigade*, and *Family Guy*; and performers including Nat and Alex Wolff and MatPat. It also covers productions such as *American Pie Presents:
Band Camp* and *
</item 8>

<item 9>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 9>

<item 10>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 10>
=== END ITEM id=c5dbf1eceb1fbdcc ===

=== ITEM id=a8d2e67acd14625b ===
Gold sentences and facts:
Sentence 0: Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist Tayeb Salih.
  Fact 0.0: Season of Migration to the North is a classic post-colonial Sudanese novel.
  Fact 0.1: Season of Migration to the North is written by the novelist Tayeb Salih.
  Fact 0.2: The Arabic title of Season of Migration to the North is موسم الهجرة إلى الشمال, transliterated Mawsim al-Hiǧra ilā ash-Shamāl.
Sentence 1:  The novel is a counter narrative to "Heart of Darkness".
  Fact 1.0: Season of Migration to the North is a counter narrative to Heart of Darkness.
Sentence 2: Heart of Darkness (1899) is a novella by Polish-British novelist Joseph Conrad, about a voyage up the Congo River into the Congo Free State, in the heart of Africa, by the story's narrator Charles Marlow.
  Fact 2.0: Heart of Darkness is dated 1899.
  Fact 2.1: Heart of Darkness is a novella.
  Fact 2.2: Heart of Darkness is by Polish-British novelist Joseph Conrad.
  Fact 2.3: Heart of Darkness is about a voyage up the Congo River into the Congo Free State.
  Fact 2.4: The voyage in Heart of Darkness goes into the heart of Africa.
  Fact 2.5: The narrator of Heart of Darkness is Charles Marlow.

Retrieved items:
<item 1>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
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
work: Season of Migration to the North
relation_to_Heart_of_Darkness: None
creator: Tayeb Salih
related_work: Heart of Darkness
related_work_creator: Joseph Conrad
Joseph_Conrad: Joseph Conrad
Conrad_relation: CREATED_BY
</item 4>

<item 5>
The Chinese Agent
The Chinese Agent (1970) is a comic novel by Michael Moorcock.

 It is a revision of "Somewhere in the Night", which Moorcock published in 1966 under the pseudonym Bill Barclay.

 Although Moorcock is best known as the author of fantasy fiction and science fiction-based parables such as "Behold the Man" and "The Dancers at the End of
Time", here he writes a light-hearted caper that parodies the spy novel genre.
</item 5>

<item 6>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 6>

<item 7>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 7>

<item 8>
First North Americans
The novels take into account new developments in North American archaeology such as the discovery of Kennewick Man and the development of the coastal-route
model as a possible alternative or supplement to overland migration across Beringia.
</item 8>

<item 9>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 9>

<item 10>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 10>
=== END ITEM id=a8d2e67acd14625b ===

=== ITEM id=b67f7d1156867d3e ===
Gold sentences and facts:
Sentence 0: The Wire Fox Terrier is a breed of dog, one of many terrier breeds.
  Fact 0.0: The Wire Fox Terrier is a breed of dog.
  Fact 0.1: The Wire Fox Terrier is one of many terrier breeds.
Sentence 1: The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.
  Fact 1.0: The West Siberian Laika, also called WSL, is a breed of hunting dog.
  Fact 1.1: The West Siberian Laika, also called WSL, is a breed of spitz type.

Retrieved items:
<item 1>
Teddy Roosevelt Terrier
The Teddy Roosevelt Terrier is a small to medium-sized American hunting terrier.

 Lower-set with shorter legs, more muscular, and heavier bone density than its cousin the American Rat Terrier.

 There is much diversity in the history of the Teddy Roosevelt Terrier breed and it shares a common early history with the American Rat Terrier, Fox Paulistinha
and Tenterfield Terrier.

 It is said the Rat Terrier background stems from the terriers or other dogs that were brought over by early English and other working class immigrants.

 Since the breed was a farm, hunting and utility dog there was little to no planned breeding other than breeding dogs with agreeable traits to each other in
order to produce the desired work ethic in the dog.

 It is assumed that the Feist (dog), Bull Terrier, Smooth Fox Terrier, Manchester Terrier, Whippet, Italian Greyhound, the now extinct English White Terrier,
Turnspit dog and or Wry Legged Terrier all share in the Teddy Roosevelt Terrier's ancestry.

 These early Ratting Terriers were then most likely bred to the Beagle or Beagle cross bred dogs (for increased scenting ability) and other dogs.
</item 1>

<item 2>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 2>

<item 3>
Wire Fox Terrier
The Wire Fox Terrier is a breed of dog, one of many terrier breeds.

 It is a fox terrier, and although it bears a resemblance to the smooth fox terrier, they are believed to have been developed separately.
</item 3>

<item 4>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 4>

<item 5>
West Siberian Laika
The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.

 Russian publications indicate that the term West Siberian Laika loosely applied to hunting dogs originating with the Mansi and Khanty people in Ural and West
Siberia, but there were no standards or registrations of WSL as such until 1930.

 Then WWll disrupted it for a while, but "systematic breeding with registrations" resumed after the war ended, in 1946.

 This was the time the breed began taking modern shape.

 Before that hunters only knew of Mansi Laika and Khanty Laika.

 In early 1960 many hunters in Ural still preferred the term Mansi Laika, when speaking of West Siberian Laika.

 In Russian language, the term Laika originated from the word "layat" that means to bark.

 The word Laika simply means "barker".

 Any hunting Laika is a bark pointer (pointing at animal of interest by barking and staying with the animal ).

 It is a versatile dog depending on use and environment, but in certain parts of the country they have become more specialized.
</item 5>

<item 6>
Old Jock
Old Jock (1859–1871), was a Fox Terrier famous during the late 19th and early 20th centuries.

 A mostly white dog, he ran briefly with a hunting kennel before becoming a show dog, most notably with a victory at the show which popularised the Fox Terrier.

 His main show rivalry was with a dog named Tartar, and along with a dog named Trap, the three were popular sires of the Fox Terrier breed.

 He was also involved in the early formation of the Jack Russell Terrier and the Dandie Dinmont Terrier breeds.
</item 6>

<item 7>
Signal Circuit of Halleston
Signal Circuit of Halleston was a Wire Fox Terrier and winner of the 1926 Westminster Kennel Club Dog Show.

 This was the 50th Westminster show, and the fourth Fox Terrier to win best in show.
</item 7>

<item 8>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 8>

<item 9>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 9>

<item 10>
Bulldog
The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.

 Other scent-hound breeds include the Small Greek Domestic Dog, Irish Wolfhound, Bluetick Coonhound, Finnish Lapphund, and the Basset Hound.

 The Bulldog is a muscular, hefty dog with a wrinkled face and a distinctive pushed-in nose.

 The American Kennel Club (AKC), The Kennel Club (UK), and the United Kennel Club (UKC) oversee breeding records.

 Bulldogs were the fourth most popular purebreed in the US in 2007 according to the American Kennel Club.
</item 10>
=== END ITEM id=b67f7d1156867d3e ===

=== ITEM id=9e3dc35cbfe5e96b ===
Gold sentences and facts:
Sentence 0: Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.
  Fact 0.0: Jelena Janković was born on 28 February 1985.
  Fact 0.1: Jelena Janković's name is written in Serbian Cyrillic as Јелена Јанковић.
  Fact 0.2: Jelena Janković is a Serbian professional tennis player.
Sentence 1: Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian politician and retired professional tennis player.
  Fact 1.0: Marat Mubinovich Safin's name is written in Russian as Мара́т Муби́нович Са́фин.
  Fact 1.1: Marat Mubinovich Safin's name is written in Tatar as Марат Мөбин улы Сафин.
  Fact 1.2: Marat Mubinovich Safin's Tatar name is romanized as Marat Möbin uğlı Safin.
  Fact 1.3: Marat Mubinovich Safin was born on 27 January 1980.
  Fact 1.4: Marat Mubinovich Safin is a Russian politician.
  Fact 1.5: Marat Mubinovich Safin is a retired professional tennis player.

Retrieved items:
<item 1>
Jelena Janković career statistics
This is a list of the main career statistics of Serbian professional tennis player and former World No. 1, Jelena Janković.

 To date, Janković has won 15 WTA singles titles including four WTA Tier I singles titles, one WTA Premier Mandatory singles title and one WTA Premier 5 singles
title.

 Janković was also the runner-up in singles at the 2008 US Open, a semi-finalist at the year-ending WTA Tour Championships in 2008, 2009 and 2013 and a
quarterfinalist at the 2008 Beijing Olympics.
</item 1>

<item 2>
2008 Wimbledon Championships – Mixed Doubles
Jamie Murray and Jelena Janković were the defending champions, but Janković chose not to participate, and only Murray competed that year.
</item 2>

<item 3>
Dinara Safina career statistics
This is a list of the main career statistics of retired, Russian professional tennis player, Dinara Safina.

 Throughout her career, Safina won twelve WTA singles titles including three Tier I singles titles at the 2008 Qatar Telecom German Open, Rogers Cup and Toray
Pan Pacific Open respectively; one Premier Mandatory singles title at the 2009 Mutua Madrileña Madrid Open and one Premier 5 singles title at the 2009
Internazionali BNL d'Italia.

 She was also the runner-up at the 2008 French Open and the 2009 Australian Open and French Open as well as a silver medalist in singles at the 2008 Beijing
Olympics.
</item 3>

<item 4>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 4>

<item 5>
Ana Ivanovic career statistics
This is a list of the main career statistics of Serbian professional tennis player, Ana Ivanovic.

 Ivanovic won fifteen WTA singles titles including one grand slam singles title at the 2008 French Open and three WTA Tier I singles titles.

 She was also the runner-up at the 2007 French Open and 2008 Australian Open and a semi-finalist at the 2007 Wimbledon Championships and 2007 WTA Tour
Championships.

 On June 9, 2008 Ivanovic became the world No. 1 for the first time in her career.
</item 5>

<item 6>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 6>

<item 7>
Jelenin svet
Jelenin svet ("Jelena's World") is a 2008 independent documentary film written and directed by Tanja Brzaković, about former World No. 1 female tennis player,
Jelena Janković.
</item 7>

<item 8>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 8>

<item 9>
Jelena Janković
Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.

 Janković is a former World No. 1 in singles, a ranking achieved preceding her finals appearance at the 2008 US Open.

 Janković's career highlights include winning the 2007 Wimbledon Mixed Doubles title with Jamie Murray, the 2010 Indian Wells Masters, the Internazionali BNL
d'Italia twice, in 2007 and 2008, and the aforementioned appearance in the finals of the 2008 US Open.

 She is coached by her brother Marko.
</item 9>

<item 10>
Ivanovic–Janković rivalry
Serbian tennis players Ana Ivanovic and Jelena Janković are both former world No. 1 and have met 12 times so far.

 Their overall head-to-head series is 9–3, in Ivanovic's favor.

 Janković is two years older than Ivanovic, so their paths connect them to play each other.

 They are the first female players to achieve major successes for independent Serbia.
</item 10>
=== END ITEM id=9e3dc35cbfe5e96b ===
