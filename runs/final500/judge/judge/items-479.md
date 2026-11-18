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

This file has 12 items.

=== ITEM id=f80403292e6ddb7c ===
Gold sentences and facts:
Sentence 0: Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
  Fact 0.0: Ruggero Deodato was born on 7 May 1939.
  Fact 0.1: Ruggero Deodato is an Italian film director.
  Fact 0.2: Ruggero Deodato is an Italian screenwriter.
  Fact 0.3: Ruggero Deodato is an Italian actor.
Sentence 1: Alejandro Springall is a Mexican film director and producer.
  Fact 1.0: Alejandro Springall is a Mexican film director.
  Fact 1.1: Alejandro Springall is a Mexican producer.

Retrieved items:
<item 1>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 1>

<item 2>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 2>

<item 3>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 3>

<item 4>
Actor John Steiner became a fixture in Italian genre films, including works by Tinto Brass. Ruggero Deodato, Italian director, writer and actor, reportedly
directed most of 1964’s horror-inflected peplum *Hercules, Prisoner of Evil* while credited assistant director Antonio Margheriti was occupied elsewhere.
Deodato’s *Cannibal Holocaust* (1980), filmed in Colombia, drew on staged-seeming news coverage and helped drive the 197
</item 4>

<item 5>
Cannibal Holocaust
Cannibal Holocaust is a 1980 Italian cannibal horror film directed by Ruggero Deodato from a screenplay by Gianfranco Clerici.

 It stars Carl Gabriel Yorke, Robert Kerman, Francesca Ciardi, Perry Pirkanen, and Luca Barbareschi.

 Influenced by the works of Mondo director Gualtiero Jacopetti, the film was inspired by Italian media reporting of Red Brigade terrorism.

 The coverage included news reports Deodato believed to be staged, an idea which became an integral aspect of the film's story.

 "Cannibal Holocaust" was filmed primarily in the Amazon rainforest of Colombia with indigenous tribes interacting with American and Italian actors.
</item 5>

<item 6>
Alejandro Springall
Alejandro Springall is a Mexican film director and producer.
</item 6>

<item 7>
The entries cover international cinema from silent films through the 1960s, including works about Verdi, Zola and Lucrezia Borgia, and films by Jacques Feyder,
Max Ophüls and Mauritz Stiller. Later highlights include John Steiner’s Italian genre films; Ruggero Deodato’s disputed role directing *Hercules, Prisoner of
Evil* and controversial *Cannibal Holocaust*; Stefan Ruzowitzky’s Oscar-winning *The Counterfeit
</item 7>

<item 8>
Ruggero Deodato
Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
</item 8>

<item 9>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 9>

<item 10>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 10>
=== END ITEM id=f80403292e6ddb7c ===

=== ITEM id=174259f882b983cd ===
Gold sentences and facts:
Sentence 0: Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.
  Fact 0.0: Eucryphia is a small genus.
  Fact 0.1: Eucryphia consists of trees and large shrubs.
  Fact 0.2: Eucryphia is native to the south temperate regions of South America.
  Fact 0.3: Eucryphia is native to coastal eastern Australia.
Sentence 1:  Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.
  Fact 1.0: Eucryphia is sometimes placed in a family of its own, the Eucryphiaceae.
  Fact 1.1: More recent classifications place Eucryphia in the Cunoniaceae.
Sentence 2:  There are seven species, two in South America and five in Australia, and several named hybrids.
  Fact 2.0: Eucryphia has seven species.
  Fact 2.1: Two Eucryphia species are in South America.
  Fact 2.2: Five Eucryphia species are in Australia.
  Fact 2.3: Eucryphia has several named hybrids.
Sentence 3: Lepidozamia is a genus of two species of cycad, both endemic to Australia.
  Fact 3.0: Lepidozamia is a genus of cycad.
  Fact 3.1: Lepidozamia has two species.
  Fact 3.2: Both Lepidozamia species are endemic to Australia.

Retrieved items:
<item 1>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 1>

<item 2>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 2>

<item 3>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 3>

<item 4>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 4>

<item 5>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 5>

<item 6>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 6>

<item 7>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 7>

<item 8>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 8>

<item 9>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 9>

<item 10>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 10>
=== END ITEM id=174259f882b983cd ===

=== ITEM id=7eac9239469fb99c ===
Gold sentences and facts:
Sentence 0: Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost figures of the French Avant-garde theatre.
  Fact 0.0: Eugène Ionesco was born Eugen Ionescu.
  Fact 0.1: Eugène Ionesco was born on 26 November 1909.
  Fact 0.2: Eugène Ionesco died on 28 March 1994.
  Fact 0.3: Eugène Ionesco was a Romanian-French playwright.
  Fact 0.4: Eugène Ionesco wrote mostly in French.
  Fact 0.5: Eugène Ionesco was one of the foremost figures of the French Avant-garde theatre.
Sentence 1: Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."
  Fact 1.0: Paul Mark Scott was born on 25 March 1920.
  Fact 1.1: Paul Mark Scott died on 1 March 1978.
  Fact 1.2: Paul Mark Scott was an English novelist.
  Fact 1.3: Paul Mark Scott was an English playwright.
  Fact 1.4: Paul Mark Scott was an English poet.
  Fact 1.5: Paul Mark Scott is best known for his monumental tetralogy "The Raj Quartet."

Retrieved items:
<item 1>
The Lesson
The Lesson (French: "La Leçon" ) is a one-act play by French-Romanian playwright Eugène Ionesco.

 It was first performed in 1951 in a production directed by Marcel Cuvelier (who also played the Professor).

 Since 1957 it has been in permanent showing at Paris' Théâtre de la Huchette, on an Ionesco double-bill with The Bald Soprano.

 The play is regarded as an important work in the "Theatre of the Absurd".
</item 1>

<item 2>
Paul Scott (novelist)
Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."

 His novel "Staying On" won the Booker Prize for 1977.
</item 2>

<item 3>
Exit the King
Exit the King (French: Le Roi se meurt ) is an absurdist drama by Eugène Ionesco that premiered in 1962.

 It is the third in Ionesco's "Berenger Cycle", preceded by "The Killer" (1958) and "Rhinocéros" (1959), and followed by "A Stroll in the Air" (1963).
</item 3>

<item 4>
Eugène Ionesco
Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost
figures of the French Avant-garde theatre.

 Beyond ridiculing the most banal situations, Ionesco's plays depict the solitude and insignificance of human existence in a tangible way.
</item 4>

<item 5>
The Killer (play)
The Killer (French: Tueur sans gages , sometimes translated "The Killer without Reason" or "The Killer without Cause") is a play written by Eugène Ionesco in
1958.

 It is the first of Ionesco's Berenger plays, the others being "Rhinocéros" (1959), "Exit the King" (1962), and "A Stroll in the Air" (1963).
</item 5>

<item 6>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 6>

<item 7>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 7>

<item 8>
The New Tenant
The New Tenant (French: "Le Nouveau Locataire" ) is a play written by Eugène Ionesco in 1955.

 The central image is common to many Ionesco plays: something accumulates on stage and overwhelms the characters.

 In this case its furniture.

 The main characters are a gentleman, a caretaker, and two movers.

 The caretaker talks as the gentleman, the "new tenant" of the title, directs the two movers who continuously bring in furniture.
</item 8>

<item 9>
Victims of Duty
Victims of Duty (French: Victimes du Devoir ) is a one-act play written in 1953 by French-Romanian playwright Eugène Ionesco.

 An early work, it has not received the notoriety of his other works.

 The play is in the Theatre of the Absurd style, of which Ionesco was a pioneer.
</item 9>

<item 10>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 10>
=== END ITEM id=7eac9239469fb99c ===

=== ITEM id=ce9166f3bb15b5f8 ===
Gold sentences and facts:
Sentence 0: War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor at Kean University, and Albie Hecht.
  Fact 0.0: War/Dance is a 2007 American documentary film.
  Fact 0.1: War/Dance was written by Sean Fine and Andrea Nix Fine.
  Fact 0.2: War/Dance was directed by Sean Fine and Andrea Nix Fine.
  Fact 0.3: War/Dance was produced by Shine Global's Susan MacLaury.
  Fact 0.4: Susan MacLaury is a professor at Kean University.
  Fact 0.5: War/Dance was produced by Albie Hecht.
Sentence 1: The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy government and Nazi Germany during World War II.
  Fact 1.0: The Sorrow and the Pity is a documentary film.
  Fact 1.1: The Sorrow and the Pity's French title is Le Chagrin et la Pitié.
  Fact 1.2: The Sorrow and the Pity is in two parts.
  Fact 1.3: The Sorrow and the Pity was released in 1969.
  Fact 1.4: The Sorrow and the Pity was made by Marcel Ophüls.
  Fact 1.5: The Sorrow and the Pity is about the collaboration between the Vichy government and Nazi Germany.
  Fact 1.6: That collaboration occurred during World War II.

Retrieved items:
<item 1>
Polio v/s Polio victims
Polio v/s Polio Victims is the National Award Winning Documentary film.

 The documentary won "Best Motivational/ Educational Film" at the 56th National Film Awards.

 The documentary follows a group of polio victims who took to the streets in 2008 to spread the message of the Pulse Polio Campaign.

 The film follows them as they go door to door, in various slums in Mumbai, to spread awareness about the upcoming "Pulse Polio Day".
</item 1>

<item 2>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 2>

<item 3>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 3>

<item 4>
The Silent Historian
The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as
part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
</item 4>

<item 5>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 5>

<item 6>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 6>

<item 7>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 7>

<item 8>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 8>

<item 9>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 9>

<item 10>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 10>
=== END ITEM id=ce9166f3bb15b5f8 ===

=== ITEM id=9664fbb13b0ee092 ===
Gold sentences and facts:
Sentence 0: The Australian women's national soccer team represents Australia in international women's soccer at the senior level.
  Fact 0.0: The Australian women's national soccer team represents Australia in international women's soccer.
  Fact 0.1: The Australian women's national soccer team represents Australia at the senior level.
Sentence 1:  The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.
  Fact 1.0: The Australian women's national soccer team's official nickname is the Matildas.
  Fact 1.1: The nickname Matildas comes from the Australian folk song "Waltzing Matilda".
  Fact 1.2: The Australian women's national soccer team was known as the Female Socceroos before 1995.
Sentence 2: "Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
  Fact 2.0: "Waltzing Matilda" is Australia's best-known bush ballad.
  Fact 2.1: "Waltzing Matilda" has been described as the country's "unofficial national anthem."

Retrieved items:
<item 1>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 1>

<item 2>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 2>

<item 3>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 3>

<item 4>
Australia (Gyroscope song)
"Australia" is a song by Australian rock band Gyroscope, from the album "Breed Obsession".

 It was released as the third single from their highly successful album, which peaked at number one on the Australian Albums Chart.

 The song was released to radio in May 2008, and was released as a CD single and digital download on 14 June 2008, along with two acoustic pieces.

 One of those being a live acoustic version of "Australia", the other a live acoustic cover of Chris Isaak's "Wicked Game".
</item 4>

<item 5>
Vanuatu national under-23 football team
The Vanuatu national under-23 football team, also known as Vanuatu Cyclone, represents Vanuatu at U23 tournaments.

 The team is considered to be the feeder team for the Vanuatu national football team.

 They are controlled by the Vanuatu Football Federation.

 The team has gained notoriety for thrashing Micronesia's side 46–0 in the 2015 Pacific Games.
</item 5>

<item 6>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 6>

<item 7>
Spencer Park, Queensland
Spencer Park (currently known as Corporate Travel Management Stadium) is a football stadium located in the suburb of Newmarket, north of Brisbane, Queensland,
Australia.

 It is home to National Premier Leagues Queensland team Brisbane City FC.

 The stadium, while officially holding 10,000, seats only 3,000 under the grandstand, the "Gino Merlo Stand".

 The stadium opened in 1963 on the site of an old Brisbane City Council waste disposal site, and underwent a major renovation in 1981 with the construction of
the Gino Merlo Stand.

 The stadium is home to La Rustica restaurant, where traditional Italian foods, including pizza, are served.
</item 7>

<item 8>
Forests of Australia
Australia has many forests of importance due to significant features, despite being one of the driest continents.

 Australia has approximately 123 million hectares of native forest, which represents about 16% of Australia's land area.

 The majority of Australia's trees are hardwoods, typically eucalypts, rather than softwoods like pine.

 While softwoods dominate some native forests, their total area is judged insufficient to constitute a major forest type in Australia's National Forest
Inventory.

 The Forests Australia website provides up-to-date information on Australia's forests.

 Detailed information on Australia's forests is available from Australia's State of the Forests Reports that are published every five years.
</item 8>

<item 9>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 9>

<item 10>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 10>
=== END ITEM id=9664fbb13b0ee092 ===

=== ITEM id=e6e52f7d64dfd74f ===
Gold sentences and facts:
Sentence 0: Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.
  Fact 0.0: Ghost Rider is a song.
  Fact 0.1: Ghost Rider is a song by the protopunk band Suicide.
  Fact 0.2: Ghost Rider appears on the debut album of the protopunk band Suicide.
Sentence 1:  The song is based on the Marvel Comics character.
  Fact 1.0: The song Ghost Rider is based on the Marvel Comics character.
Sentence 2: Suicide is the debut album from the American rock band Suicide.
  Fact 2.0: The album Suicide is the debut album of the American rock band Suicide.
  Fact 2.1: The band Suicide is an American rock band.
Sentence 3:  It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.
  Fact 3.0: The album Suicide was released in 1977.
  Fact 3.1: The album Suicide was released on Red Star Records.
  Fact 3.2: The album Suicide was produced by Craig Leon.
  Fact 3.3: The album Suicide was produced by Marty Thau.

Retrieved items:
<item 1>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 1>

<item 2>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 2>

<item 3>
Ghost Rider (Suicide song)
Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.

 The song is based on the Marvel Comics character.
</item 3>

<item 4>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 4>

<item 5>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 5>

<item 6>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 6>

<item 7>
Radio Blast
Radio Blast is the fifth E.P. by Chicago-based pop punk band Screeching Weasel.

 The lineup for this single is the same as their 1993 album, Wiggle.

 2000 copies were initially pressed on Underdog Records.

 A later repress came out on burgundy colored vinyl and had a sticker on the sleeve that proclaimed it "not part of the limited edition".

 Both pressings are now out of print, but the two songs from this E.P. can be found on the B-sides and rarities collection, Kill the Musicians.

 The A-side is a vicious attack on the state of rock radio.

 Side B is the sad story of a girl who cannot find love and ends up committing suicide.

 "The Girl Next Door" was covered by Blink-182, which was first released on a 1998 reissue of their first demo album, "Buddha".
</item 7>

<item 8>
The text surveys albums, songs, bands, and labels across punk, alternative, electronic, dance, worship, and other genres. Highlights include Mucky Pup’s 1989
album; Suicide’s 1977 debut and “Ghost Rider”; influential alternative acts such as the Replacements and R.E.M.; and varied artists from Beasts of Bourbon to
the Shins. It also profiles labels including Manimal Vinyl, Kinetic, and Sargent House, alongside numerous smaller or short-lived bands and
</item 8>

<item 9>
- **Quiet Riot 10** (2014), the band’s twelfth studio album and first since reuniting in 2010, features Jizzy Pearl on six new studio tracks; its final four
tracks are live performances with Kevin DuBrow, who died in 2007.
- **Inner Secrets** (1978), Santana’s ninth album, shifted toward album-oriented rock. “Stormy” and “One Chain” were hits; “Well All Right” reached No.
</item 9>

<item 10>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 10>
=== END ITEM id=e6e52f7d64dfd74f ===

=== ITEM id=2ceca2185110ec61 ===
Gold sentences and facts:
Sentence 0:  It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of Superman called Nuclear Man.
  Fact 0.0: The film is the fourth and final film in the original Superman film series.
  Fact 0.1: The film is the first film in the original Superman series not produced by Alexander and Ilya Salkind.
  Fact 0.2: The film was produced by Golan-Globus' Cannon Films.
  Fact 0.3: The film was produced in association with Warner Bros.
  Fact 0.4: Gene Hackman returned as Lex Luthor in the film.
  Fact 0.5: Lex Luthor creates an evil solar-powered version of Superman called Nuclear Man.
Sentence 1: Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.
  Fact 1.0: Eugene Allen 'Gene' Hackman was born on January 30, 1930.
  Fact 1.1: Gene Hackman is a retired American actor.
  Fact 1.2: Gene Hackman is a retired American novelist.

Retrieved items:
<item 1>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 1>

<item 2>
Lex Luthor (Smallville)
Lex Luthor is a fictional character from the television series "Smallville".

 He was a series regular from the pilot episode until the season seven finale, and has been played continuously by Michael Rosenbaum, with various actors
portraying the character as a child throughout the series.

 The character of Lex Luthor, first created for comic books by Jerry Siegel and Joe Shuster in 1940 as nemesis of Superman, was adapted to television in 2001 by
Alfred Gough and Miles Millar—this is only the third time the character has been adapted to a live action television series.

 The character has also appeared in various literature based on the "Smallville" television series, none of which directly continues from or into the television
episodes.
</item 2>

<item 3>
Lex Luthor: Man of Steel
Lex Luthor: Man of Steel (later collected as simply Luthor) is a five-issue monthly American comic book limited series written by Brian Azzarello and
illustrated by Lee Bermejo, which features Superman's nemesis Lex Luthor as the protagonist.
</item 3>

<item 4>
It's Superman!
It's Superman!

 is a novel by Tom De Haven based on the comic book superhero Superman.

 It was released on September 15, 2005 in hardcover and August 29, 2006 in paperback.

 The premise tells the tale of Clark Kent's beginnings into becoming Superman, set in the 1930s, where Clark befriends a wrongly convicted photographer named
Willi Berg, and is then taken from Kansas to Hollywood and finally in New York where he meets Lois Lane, fights Lex Luthor, as he debuts in his superhero
persona.

 Despite the setting, this is not about the Golden Age Superman also known as the Superman of Earth-2; as Perry White, the Daily Planet, Lex Luthor's position
and his trademark powers (including flight) are not part of that alternate Earth.

 Rather, it's a Superman period piece set in the 1930s.
</item 4>

<item 5>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 5>

<item 6>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 6>

<item 7>
Lionel Luthor
Lionel Luthor is a fictional character portrayed by John Glover in the television series "Smallville".

 The character was initially a recurring guest in season one, and became a series regular in season two and continued until being written out of the show in
season seven.

 The character returned to the show in season ten again as a recurring guest role as a parallel universe (Earth-2) version of the character.

 In "Smallville", Lionel Luthor is the father of Lex Luthor (Michael Rosenbaum), and founder and CEO of LuthorCorp.

 Lex Luthor's father was first introduced in "Superman" comics by Jerry Siegel in 1961 and has since appeared in other Superman-related media under different
names.

 "Smallville" is the first appearance in which the character has been an intricate part of a Superman adaptation.

 Series developers Alfred Gough and Miles Millar created Lionel Luthor for "Smallville" to provide an antithesis to the parenting style of Jonathan Kent (John
Schneider) and Martha Kent (Annette O'Toole).
</item 7>

<item 8>
Superman IV: The Quest for Peace
Superman IV: The Quest for Peace is a 1987 American superhero film directed by Sidney J. Furie, based on the DC Comics character Superman.

 It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but
rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of
Superman called Nuclear Man.
</item 8>

<item 9>
Alternative versions of Lex Luthor
The fictional character Lex Luthor has appeared in a number of media, always as the archenemy of Superman.

 Each version of the work typically establishes its own continuity, and sometimes introduces parallel universes, to the point where distinct differences in the
portrayal of the character can be identified.

 In addition, the DC Universe has been rewritten a number of times, establishing additional versions of the character.

 This article details and lists various versions of Lex Luthor depicted in works including DC Comics "Multiverse", "Elseworlds", television and film.
</item 9>

<item 10>
Batman v Superman: Dawn of Justice
Batman v Superman: Dawn of Justice is a 2016 American superhero film featuring the DC Comics characters Batman and Superman.

 Directed by Zack Snyder and distributed by Warner Bros.

 Pictures, the film is the second installment in the DC Extended Universe (DCEU) following 2013's "Man of Steel".

 It was written by Chris Terrio and David S. Goyer, and features an ensemble cast that includes Ben Affleck, Henry Cavill, Amy Adams, Jesse Eisenberg, Diane
Lane, Laurence Fishburne, Jeremy Irons, Holly Hunter and Gal Gadot.

 "Batman v Superman: Dawn of Justice" is the first live-action film to feature Batman and Superman together, as well as the first live-action cinematic
portrayal of Wonder Woman, Aquaman, Flash and Cyborg.

 In the film, criminal mastermind Lex Luthor manipulates Batman into a preemptive battle with Superman, whom Luthor is obsessed with defeating.
</item 10>
=== END ITEM id=2ceca2185110ec61 ===

=== ITEM id=03c94c0944567811 ===
Gold sentences and facts:
Sentence 0: 888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.
  Fact 0.0: 888 7th Avenue is a 628 ft (191 m) tall modern-style office skyscraper.
  Fact 0.1: 888 7th Avenue is in Midtown Manhattan.
  Fact 0.2: 888 7th Avenue was completed in 1969.
  Fact 0.3: 888 7th Avenue has 46 floors.
Sentence 1: 200 West Street is the global headquarters of the Goldman Sachs investment banking firm.
  Fact 1.0: 200 West Street is the global headquarters of the Goldman Sachs investment banking firm.
Sentence 2:  The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.
  Fact 2.0: The building is 749 ft tall.
  Fact 2.1: The building has 44 stories.
  Fact 2.2: The building is located on West Street.
  Fact 2.3: The building is located between Vesey Street and Murray Street.
  Fact 2.4: The building is in Lower Manhattan.

Retrieved items:
<item 1>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 1>

<item 2>
West Street District
The West Street District is a historic district on West Street in Boston, Massachusetts, one of the city's "ladder districts" that runs between Tremont Street
and Washington Street in the Downtown Crossing commercial/retail area.

 The district includes four buildings located near the corner of Tremont and West Streets, all built in the early 20th century.

 The two buildings at 148-150 Tremont Street were once occupied by Chandler and Company, an exclusive department store.

 Number 148 is a Renaissance Revival structure built as an office building in 1912, and number 150 was built in 1903 to house the Oliver Ditson Company, a music
publisher.

 The Fabyan building at 26-30 West Street was designed by Coolidge, Shepley, Bulfinch & Abbott, and built in 1926.

 The Schraffts Building at 16-24 West Street was built in 1922, and housed a flagship candy store and restaurant for more than fifty years.
</item 2>

<item 3>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 3>

<item 4>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 4>

<item 5>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 5>

<item 6>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
</item 6>

<item 7>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 7>

<item 8>
Pershing Square Capital Management
Pershing Square Capital Management is an American hedge fund management company founded and run by Bill Ackman, located at 888 7th Avenue in New York.
</item 8>

<item 9>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 9>

<item 10>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 10>
=== END ITEM id=03c94c0944567811 ===

=== ITEM id=2e88f3e812a09c1c ===
Gold sentences and facts:
Sentence 0:  He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
  Fact 0.0: The person referred to as 'He' has had a cameo acting role as a tailor in The Wolf of Wall Street.
  Fact 0.1: The person referred to as 'He' has had a cameo acting role as a tailor in The Good Shepherd.
Sentence 1: The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir of the same name by Jordan Belfort.
  Fact 1.0: The Wolf of Wall Street is a 2013 film.
  Fact 1.1: The Wolf of Wall Street is American.
  Fact 1.2: The Wolf of Wall Street is a biographical black comedy crime film.
  Fact 1.3: The Wolf of Wall Street was directed by Martin Scorsese.
  Fact 1.4: The Wolf of Wall Street was written by Terence Winter.
  Fact 1.5: The Wolf of Wall Street is based on the memoir of the same name by Jordan Belfort.

Retrieved items:
<item 1>
The Wolf of Wall Street (2013 film)
The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir
of the same name by Jordan Belfort.

 It recounts Belfort's perspective on his career as a stockbroker in New York City and how his firm Stratton Oakmont engaged in rampant corruption and fraud on
Wall Street that ultimately led to his downfall.

 Leonardo DiCaprio (who was also a producer) stars as Belfort, with Jonah Hill as his business partner and friend Donnie Azoff, Margot Robbie as his wife Naomi
Lapaglia and Kyle Chandler as Patrick Denham, the FBI agent who tries to bring him down.

 Matthew McConaughey, Rob Reiner, Jon Favreau, Joanna Lumley and Jean Dujardin also star.
</item 1>

<item 2>
Leonard Logsdail
Leonard Logsdail (born September 11, 1950 in London, England) is a bespoke tailor, based in Manhattan, New York, specializing in men's suits.

 He is described as one of the finest bespoke tailors in the men’s suit business.

 All of Logsdail's suits are crafted and perfected on-site in his New York City location.

 He is credited for making Savile Row tailoring a local option in New York.

 Logsdail is known for creating high-end suits, including lining jackets with Hermes silk scarves.

 CNBC talk show host, economist and fashion icon Larry Kudlow has his suits made by Leonard Logsdail.

 Logsdail has created suits for award-winning films, and is recognized as one of cinema's most sought-after tailors.

 Logsdail has collaborated with esteemed film directors Steven Spielberg, Robert de Niro, Oliver Stone, Ridley Scott, and Martin Scorcese.

 He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
</item 2>

<item 3>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 3>

<item 4>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 4>

<item 5>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 5>

<item 6>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 6>

<item 7>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 7>

<item 8>
The Wolf of Wall Street (2013 film)
The film marks the director's fifth collaboration with DiCaprio, after "Gangs of New York" (2002), "The Aviator" (2004), "The Departed" (2006) and "Shutter
Island" (2010), as well as his second collaboration with Winter after the television series "Boardwalk Empire" (2010–14).
</item 8>

<item 9>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 9>

<item 10>
A Walk in the Woods (film)
A Walk in the Woods is a 2015 American biographical comedy-drama film directed by Ken Kwapis, based on the 1998 book/memoir of the same name by Bill Bryson.

 The film stars Robert Redford, Nick Nolte and Emma Thompson.

 The film was released on September 2, 2015, by Broad Green Pictures.
</item 10>
=== END ITEM id=2e88f3e812a09c1c ===

=== ITEM id=76d65e168af1c9ae ===
Gold sentences and facts:
Sentence 0: Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also in Halland.
  Fact 0.0: Spettekaka is also called spettkaka.
  Fact 0.1: In native Scanian, spettekaka is called "spiddekaga".
  Fact 0.2: Spettekaka is a local dessert.
  Fact 0.3: Spettekaka is a local dessert of the southern parts of Sweden.
  Fact 0.4: Spettekaka is chiefly found in the province of Scania (Skåne).
  Fact 0.5: Spettekaka is also found in Halland.
Sentence 1: Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.
  Fact 1.0: Scanian is a closely related group of South Swedish dialects.
  Fact 1.1: Scanian dialects are spoken in the province of Scania.
  Fact 1.2: Scanian dialects are spoken in southern Sweden.
Sentence 2:  Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.0: Scanian formed part of the old Scandinavian dialect continuum.
  Fact 2.1: Most historical linguists consider Scanian to be an East Danish dialect group.
  Fact 2.2: Standard Swedish has influenced the Scanian dialects in the region in the modern era.
  Fact 2.3: The Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.4: The Scanian dialects have been treated as South Swedish due to the modern-era influence from Standard Swedish in the region.
  Fact 2.5: The Scanian dialects have been treated as South Swedish because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders.

Retrieved items:
<item 1>
Scanian dialect
Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.

 Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to
the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered
isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.

 However, many of the early Scandinavian linguists, including Adolf Noreen and G. Sjöstedt, classified it as "South Scandinavian", and some linguists, such as
Elias Wessén, also considered Old Scanian a separate language, classified apart from both Old Danish and Old Swedish.
</item 1>

<item 2>
The Saga of Gosta Berling
The Saga of Gosta Berling (Swedish: Gösta Berlings saga ) is a 1924 Swedish romantic drama film directed by Mauritz Stiller and released by AB Svensk
Filmindustri, starring Lars Hanson, Gerda Lundequist and Greta Garbo in her native break-out role on film.

 The film is based on the 1891 debut novel of the same name by the Swedish author and Nobel Prize winner Selma Lagerlöf.

 It is also known as Gosta Berling's Saga, The Story of Gosta Berling and The Atonement of Gosta Berling.
</item 2>

<item 3>
Skogsrå
The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical
creature of the forest, or Rå, in Swedish folklore.
</item 3>

<item 4>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 4>

<item 5>
Hakata dialect
Hakata dialect (博多弁 , Hakata-ben ) is a Japanese dialect spoken in Fukuoka city.

 Hakata dialect originated in Hakata commercial district, while a related Fukuoka dialect (福岡弁 , Fukuoka-ben ) was spoken in the central district.

 Hakata dialect has spread throughout the city and its suburbs.

 Most of Japanese regard Hakata dialect as the dialect typical of Fukuoka, so it is sometimes called "Fukuoka-ben".
</item 5>

<item 6>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 6>

<item 7>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 7>

<item 8>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 8>

<item 9>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 9>

<item 10>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 10>
=== END ITEM id=76d65e168af1c9ae ===

=== ITEM id=c79891a0468592aa ===
Gold sentences and facts:
Sentence 0: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 0.0: Henry Louis Mencken lived from September 12, 1880 to January 29, 1956.
  Fact 0.1: Henry Louis Mencken was an American journalist.
  Fact 0.2: Henry Louis Mencken was an American satirist.
  Fact 0.3: Henry Louis Mencken was an American cultural critic.
  Fact 0.4: Henry Louis Mencken was a scholar of American English.
Sentence 1: Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.
  Fact 1.0: Albert Camus lived from 7 November 1913 to 4 January 1960.
  Fact 1.1: Albert Camus was a French philosopher.
  Fact 1.2: Albert Camus was an author.
  Fact 1.3: Albert Camus was a journalist.

Retrieved items:
<item 1>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 1>

<item 2>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 2>

<item 3>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 3>

<item 4>
The Guest
"The Guest" (French: "L'Hôte" ) is a short story by the French writer Albert Camus.

 It was first published in 1957 as part of a collection entitled "Exile and the Kingdom" ("L'exil et le royaume").

 The French title "L'Hôte" translates into both "the guest" and "the host" which ties back to the relationship between the main characters of the story.

 Camus employs this short tale to reflect upon issues raised by the political situation in French North Africa.

 In particular, he explores the problem of refusing to take sides in the colonial conflict in Algeria, something that mirrors Camus' own non-aligned stance
which he had set out in his Nobel Prize acceptance speech.
</item 4>

<item 5>
Lois Romano
Lois Romano is an American national journalist who was an editor, reporter and columnist for The Washington Post.

 She was most recently the editor of Washington Post Live, the news organization's editorial events business.

 In early 2015, she returned to the newspaper where she had a long career as a political correspondent and profile writer.

 During her tenure at PostLive, she raised the profile of the platform by linking it to the outlet's award-winning enterprise journalism and utilizing its deep
bench of seasoned journalists as moderators.

 She also created the high value editorial content, such as the successful “Pre-Game” series before presidential debates, a news-driven morning event called
Coffee@WaPo, and the Security Tomorrow conversations with columnist David Ignatius.

 Working with engineers, Romano helped develop the platform's first App.

 Working with newsroom leadership in 2016, Romano also ensured that The Post had its first-ever political convention venues outside the official workspaces in
both Cleveland and Philadelphia.
</item 5>

<item 6>
Michel Foucault
Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist,
and literary critic.
</item 6>

<item 7>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 7>

<item 8>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 8>

<item 9>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 9>

<item 10>
Reflections on the Guillotine
"Reflections on the Guillotine" is an extended essay written in 1957 by Albert Camus.

 In the essay Camus takes an uncompromising position for the abolition of the death penalty.

 Camus's view is similar to that of Cesare Beccaria and the Marquis de Sade, the latter having also argued that murder premeditated and carried out by the state
was the worst kind.

 Camus states that he does not base his argument on sympathy for the convicted but on logical grounds and on proven statistics.

 Camus also argues that capital punishment is an easy option for the government where remedy and reform may be possible.
</item 10>
=== END ITEM id=c79891a0468592aa ===

=== ITEM id=2b9aa87a243a3e29 ===
Gold sentences and facts:
Sentence 0: While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and Zoot Sims.
  Fact 0.0: The musician studied at the University of Michigan.
  Fact 0.1: While studying at the University of Michigan, the musician played with visiting musician Dizzy Gillespie.
  Fact 0.2: While studying at the University of Michigan, the musician played with visiting musician Miles Davis.
  Fact 0.3: While studying at the University of Michigan, the musician played with visiting musician Thad Jones.
  Fact 0.4: While studying at the University of Michigan, the musician played with visiting musician Billy Mitchell.
  Fact 0.5: After that, the musician played with Sonny Stitt, Count Basie and Al McKibbon.
  Fact 0.6: After that, the musician played with Cannonball Adderley, Percy Heath, Philly Joe Jones and Lou Donaldson.
  Fact 0.7: After that, the musician played with Donald Byrd, Kenny Dorham and Zoot Sims.
Sentence 1: Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.
  Fact 1.0: Miles Dewey Davis III was born on May 26, 1926.
  Fact 1.1: Miles Dewey Davis III died on September 28, 1991.
  Fact 1.2: Miles Dewey Davis III was an American jazz trumpeter.
  Fact 1.3: Miles Dewey Davis III was an American jazz bandleader.
  Fact 1.4: Miles Dewey Davis III was an American jazz composer.

Retrieved items:
<item 1>
Miles Davis
Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.

 He is among the most influential and acclaimed figures in the history of jazz and 20th century music.

 Davis adopted a variety of musical directions in his five-decade career which kept him at the forefront of a number of major stylistic developments in jazz.
</item 1>

<item 2>
Marty Marsala
Marty Marsala (2 April 1909 – 27 April 1975) was an American jazz trumpeter born in Chicago, perhaps best known for working from 1926-1946 with his brother Joe
Marsala in a big band in New York City and Chicago.

 He had also toured with various artists, such as Chico Marx and Miff Mole, to name a few.

 During the 1940s Marsala was a celebrated West Coast jazz trumpeter, commuting back and forth from Chicago to San Francisco frequently.

 In various club settings Marsala shared stages with Earl Hines and Sidney Bechet.
</item 2>

<item 3>
Don Ellis
Donald Johnson Ellis (July 25, 1934 – December 17, 1978) was an American jazz trumpeter, drummer, composer, and bandleader.

 He is best known for his extensive musical experimentation, particularly in the area of time signatures. Later in his life he worked as a film composer,
contributing a score to 1971's "The French Connection" and 1973's "The Seven-Ups".
</item 3>

<item 4>
Paul Smith (defensive end)
Paul Edward Smith (August 13, 1945 – March 14, 2000) was a collegiate and professional American football defensive end.

 Smith was selected in the 1968 Common Draft by the American Football League's Denver Broncos and played in the National Football League for the Broncos and
Washington Redskins.
</item 4>

<item 5>
Count Basie
William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.

 His mother taught him to play the piano and he started performing in his teens.

 Dropping out of school, he learned to operate lights for vaudeville and to improvise accompaniment for silent films at a local movie theater in his home town
of Red Bank, New Jersey.

 By age 16, he increasingly played jazz piano at parties, resorts and other venues.

 In 1924, he went to Harlem, where his performing career expanded; he toured with groups to the major jazz cities of Chicago, St. Louis and Kansas City.

 In 1929 he joined Bennie Moten's band in Kansas City, and played with them until Moten's death in 1935.
</item 5>

<item 6>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 6>

<item 7>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 7>

<item 8>
Louis Smith (gymnast)
Louis Antoine Smith, MBE (born 22 April 1989) is a British artistic gymnast.

 He received a bronze medal and a silver medal on the pommel horse at the 2008 Beijing Olympics and the 2012 London Olympics respectively, with the former
marking the first time a British gymnast had placed in an Olympic event since 1928.

 He followed this up with a second consecutive silver medal on the pommel horse at the 2016 Rio Olympics, finishing behind teammate Max Whitlock.

 Smith was part of the Great Britain team that took the bronze in the men's artistic team all-around at the 2012 London Olympics.

 He is the only British gymnast to win Olympic medals in three separate Games.

 In 2015 he became the European champion on Pommel Horse.
</item 8>

<item 9>
Thad Jones
Thaddeus Joseph Jones (March 28, 1923 – August 20, 1986) was an American jazz trumpeter, composer, and bandleader who has been called "one of the all-time
greatest jazz trumpet soloists."
</item 9>

<item 10>
Louis Smith (musician)
While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before
going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and
Zoot Sims.
</item 10>
=== END ITEM id=2b9aa87a243a3e29 ===
