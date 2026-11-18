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

=== ITEM id=bda0784d78935b1b ===
Gold sentences and facts:
Sentence 0: The Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take ) retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority ; or a substantial and specific danger to public health or safety .
  Fact 0.0: The Whistleblower Protection Act of 1989 is Pub. L. 101-12, as amended.
  Fact 0.1: The Whistleblower Protection Act of 1989 is a United States federal law.
  Fact 0.2: The Whistleblower Protection Act protects federal whistleblowers who work for the government and report agency misconduct.
  Fact 0.3: A federal agency violates the Whistleblower Protection Act if agency authorities take, or threaten to take, retaliatory personnel action against any employee or applicant.
  Fact 0.4: The retaliatory personnel action must be because of disclosure of information by that employee or applicant.
  Fact 0.5: Whistleblowers may file complaints that they believe reasonably evidence a violation of a law, rule or regulation.
  Fact 0.6: Whistleblowers may file complaints that they believe reasonably evidence gross mismanagement.
  Fact 0.7: Whistleblowers may file complaints that they believe reasonably evidence gross waste of funds.
  Fact 0.8: Whistleblowers may file complaints that they believe reasonably evidence an abuse of authority.
  Fact 0.9: Whistleblowers may file complaints that they believe reasonably evidence a substantial and specific danger to public health or safety.

Retrieved items:
<item 1>
Whistleblower Protection Act
Specifically , the Federal Circuit has wrongly accorded a narrow definition to the type of disclosure that qualifies for whistleblower protection . Additionally
, the lack of remedies under current law for most whistleblowers in the intelligence community and for whistleblowers who face retaliation in the form of
withdrawal of the employee 's security clearance leaves unprotected those who are in a position to disclose wrongdoing that directly affects our national
security . '' S. 743 would address these problems by restoring the original congressional intent of the WPA to adequately protect whistleblowers , by
strengthening the WPA , and by creating new whistleblower protections for intelligence employees and new protections for employees whose security clearance is
withdrawn in retaliation for having made legitimate whistleblower disclosures . S. 743 ultimately became Pub. L. 112 - 199 ( S. Rep. 112 - 155 ) . Related
legislation ( edit ) This section needs to be updated . Please update this article to reflect recent events or newly available information . ( January 2017 ) On
July 14 , 2014 , the United States House of Representatives voted to pass the All Circuit Review Extension Act ( H.R.
</item 1>

<item 2>
Whistleblower Protection Act
Contents ( hide ) 1 Authorized Federal Agencies 2 Legal Cases 3 Whistleblower Protection Enhancement Act and Presidential Policy Directive 19 4 Related
legislation 5 See also 6 References 7 External links Authorized federal agencies ( edit ) The Office of Special Counsel investigates federal whistleblower
complaints . In October 2008 , then - special counsel Scott Bloch resigned amid an FBI investigation into whether he obstructed justice by illegally deleting
computer files following complaints that he had retaliated against employees who disagreed with his policies . Then - Senator Barack Obama made a campaign vow
to appoint a special counsel committed to whistleblower rights . It was not until April 2011 that President Obama 's appointee Carolyn Lerner was confirmed by
the Senate . Today , the primary mission of OSC is to safeguard the merit system by protecting federal employees and applicants from prohibited personnel
practices , especially reprisal for whistleblowing . The Merit Systems Protection Board , a quasi-judicial agency that adjudicates whistleblower complaints ,
uses appointed administrative law judges who often back the government . Since 2000 , the board has ruled for whistleblowers just three times in 56 cases
decided on their merits , according to a Government Accountability Project analysis .
</item 2>

<item 3>
Whistleblower Protection Act
November 7 , 2014 . Retrieved April 27 , 2016 . Jump up ^ https://obamawhitehouse.archives.gov/sites/default/files/image/ppd-19.pdf Jump up ^ WHISTLEBLOWER
PROTECTION ENHANCEMENT ACT OF 2012 REPORT OF THE COMMITTEE ON HOMELAND SECURITY AND GOVERNMENTAL AFFAIRS , UNITED STATES SENATE TO ACCOMPANY S. 743 . April 19 ,
2012 Jump up ^ https://www.gpo.gov/fdsys/pkg/CRPT-112srpt155/pdf/CRPT-112srpt155.pdf Jump up ^ `` H.R. 4197 - Summary '' . United States Congress . Retrieved
July 15 , 2014 . Jump up ^ `` H.R. 4197 - All Actions '' . United States Congress . Retrieved July 15 , 2014 . Jump up ^ Joe Davidson . How FBI blocks
whistleblower fighting dismissal ; new bill could help others . Washington Post 2 Feb 2016 .
</item 3>

<item 4>
Whistleblower Protection Act
Washington Post 2 Feb 2016 . Jump up ^ Grassley - Leahy Bill Improves Disclosure & Review Process for FBI Whistleblowers . Website of Senator Grassley . Dec 10
, 2015 External links ( edit ) `` Whistleblower Protection Act Information . '' U.S. Securities and Exchange Commission . January 30 , 2004 . Retrieved May 3 ,
2007 OSC Form 11 for filing a whistleblower retaliation complaint . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Whistleblower_Protection_Act&oldid=838141336 '' Categories : 1989 in law 101st United States Congress United States
federal criminal legislation Whistleblower protection legislation United States federal labor legislation Hidden categories : Wikipedia articles in need of
updating from January 2017 All Wikipedia articles in need of updating All articles with vague or ambiguous time Vague or ambiguous time from January 2017 Pages
using div col without cols and colwidth parameters Talk Contents About Wikipedia Add links This page was last edited on 25 April 2018 , at 05 : 17 . About
Wikipedia
</item 4>

<item 5>
Whistleblower Protection Act
See also ( edit ) Whistleblower protection in United States False Claims Act Federal crime Immunity from prosecution Informant List of whistleblowers Qui tam
Testimony Turn state 's evidence White collar crime Witness Witness intimidation United States Federal Witness Protection Program United States Marshals Service
References ( edit ) Jump up ^ Public Law 101 - 12 ( 1989 ) at gpo.gov Jump up ^ Peter Eisler ( 2010 - 03 - 15 ) . `` Whistleblowers '' . USA TODAY . Jump up ^
High Court Trims Whistleblower Rights Jump up ^ McCarthy , Robert J. `` Blowing in the Wind : Answers for Federal Whistleblowers '' . William & Mary Policy
Review 3 : 184 ( 2012 ) . Jump up ^ `` Know Your Rights FAQ '' . National Whistleblowers Center . Retrieved April 27 , 2016 . Jump up ^ `` Whistleblower
protection case reaches US Supreme Court '' . Courage Foundation . November 7 , 2014 . Retrieved April 27 , 2016 .
</item 5>

<item 6>
Whistleblower Protection Act
4197 ; 113th Congress ) , a bill that gives authority to federal employees who want to appeal their judgment to any federal court , and which allows
whistleblowers to appeal to any U.S. Court of Appeals that has jurisdiction . The bill would extend from three years after the effective date of the
Whistleblower Protection Enhancement Act of 2012 ( i.e. , December 27 , 2012 ) , the period allowed for : ( 1 ) filing a petition for judicial review of Merit
Systems Protection Board decisions in whistleblower cases , and ( 2 ) any review of such a decision by the Director of the Office of Personnel Management ( OPM
) . There is also a bill in its early stages called the FBI Whistleblower Protection Enhancement Act . This bill was introduced by Senators Grassley and Leahy
in 2015 . It would protect whistleblowers in the FBI more completely from reprisals by supervisors , by assigning administrative law judges to adjudicate the
cases , and enabling FBI employees to appeal their decisions to the courts . This bill is still in the process of possibly getting passed and has individuals
cosponsoring it in March 2016 .
</item 6>

<item 7>
Whistleblower Protection Act
MacLean won the case in a ruling of 7 -- 2 in the Supreme Court in January 2015 . Whistleblower Protection Enhancement Act and Presidential Policy Directive 19
( edit ) President Barack Obama issued Presidential Policy Directive 19 ( PPD - 19 ) , entitled `` Protecting Whistleblowers with Access to Classified
Information '' . According to the law signed by Obama on October 10 , 2012 , it is written that `` this Presidential Policy Directive ensures that employees ( 1
) serving in the Intelligence Community or ( 2 ) who are eligible for access to classified information can effectively report waste , fraud , and abuse while
protecting classified national security information . It prohibits retaliation against employees for reporting waste , fraud , and abuse . However , according
to a report that the Committee on Homeland Security and Governmental Affairs submitted to accompany S. 743 , `` the federal whistleblowers have seen their
protections diminish in recent years , largely as a result of a series of decisions by the United States Court of Appeals for the Federal Circuit , which has
exclusive jurisdiction over many cases brought under the Whistleblower Protection Act ( WPA ) .
</item 7>

<item 8>
Whistleblower Protection Act
Whistleblower Protection Act - Wikipedia Whistleblower Protection Act Jump to : navigation , search For the Act in India , see Whistle Blowers Protection Act ,
2011 . Whistleblower Protection Act Long title An Act to amend title 5 , United States Code , to strengthen the protections available to Federal employees
against prohibited personnel practices , and for other purposes . Nicknames Whistleblower Protection Act of 1989 Enacted by the 101st United States Congress
Effective April 10 , 1989 Citations Public law 101 - 12 Statutes at Large 103 Stat. 16 Codification Titles amended 5 U.S.C. : Government Organization and
Employees U.S.C. sections amended 5 U.S.C. ch. 12 § 1201 et seq . Legislative history Introduced in the Senate as S.
</item 8>

<item 9>
Whistleblower Protection Act
Legislative history Introduced in the Senate as S. 20 by Carl Levin ( D -- MI ) on January 25 , 1989 Passed the Senate on March 16 , 1989 ( 97 - 0 , Roll call
vote 24 , via Senate.gov ) Passed the House on March 21 , 1989 ( Agreed voice vote ) Signed into law by President George H.W. Bush on April 10 , 1989 The
Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the
government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take )
retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file
complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority
; or a substantial and specific danger to public health or safety .
</item 9>

<item 10>
Whistleblower Protection Act
The most common characteristics for a court claim that are encompassed within the protection of the Act include : that the plaintiff is an employee or person
covered under the specific statutory or common law relied upon for action , that the defendant is an employer or person covered under the specific statutory or
common law relied upon for the action , that the plaintiff engaged in protected whistleblower activity , that the defendant knew or had knowledge that the
plaintiff engaged in such activity , that there was retaliatory action taken against the one doing the whistleblowing and that the unfair treatment would not
have occurred if the plaintiff had n't brought to attention the activities . Robert MacLean blew the whistle on the fact that the TSA had cut its funding for
more air marshals . In 2009 MacLean , represented by the Government Accountability Project , challenged his dismissal at the Merit Systems Protection Board , on
the grounds that `` his disclosure of the text message was protected under the Whistleblower Protection Act of 1989 , because he ' reasonably believe ( d ) '
that the leaked information disclosed ' a substantial and specific danger to public health or safety ' . ''
</item 10>
=== END ITEM id=bda0784d78935b1b ===

=== ITEM id=cf763d122ba9d969 ===
Gold sentences and facts:
Sentence 0: In the light - dependent reactions ( mainly three vital reaction steps : ( I ) Hill and Bendall 's Z - scheme of cyclic and non-cyclic reactions , ( II ) Mitchell 's chemi - osmosis for ATP synthesis and ( III ) Dolai 's S - state diagrams in water - splitting ) , one molecule of the pigment chlorophyll absorbs one photon and loses one electron . This electron is passed to a modified form of chlorophyll called pheophytin , which passes the electron to a quinone molecule , starting the flow of electrons down an electron transport chain that leads to the ultimate reduction of NADP to NADPH . In addition , this creates a proton gradient ( energy gradient ) across the chloroplast membrane , which is used by ATP synthase in the synthesis of ATP . The chlorophyll molecule ultimately regains the electron it lost when a water molecule is split in a process called photolysis , which releases a dioxygen ( O ) molecule as a waste product .
  Fact 0.0: The light-dependent reactions have three vital reaction steps.
  Fact 0.1: Step I of the light-dependent reactions is Hill and Bendall's Z-scheme of cyclic and non-cyclic reactions.
  Fact 0.2: Step II of the light-dependent reactions is Mitchell's chemi-osmosis for ATP synthesis.
  Fact 0.3: Step III of the light-dependent reactions is Dolai's S-state diagrams in water-splitting.
  Fact 0.4: In the light-dependent reactions, one molecule of the pigment chlorophyll absorbs one photon.
  Fact 0.5: In the light-dependent reactions, one molecule of chlorophyll loses one electron.
  Fact 0.6: The lost electron is passed to a modified form of chlorophyll called pheophytin.
  Fact 0.7: Pheophytin passes the electron to a quinone molecule.
  Fact 0.8: This starts the flow of electrons down an electron transport chain.
  Fact 0.9: The electron transport chain leads to the ultimate reduction of NADP to NADPH.
  Fact 0.10: The electron flow creates a proton gradient (energy gradient) across the chloroplast membrane.
  Fact 0.11: ATP synthase uses the proton gradient in the synthesis of ATP.
  Fact 0.12: The chlorophyll molecule ultimately regains the electron it lost.
  Fact 0.13: The electron is regained when a water molecule is split in a process called photolysis.
  Fact 0.14: Photolysis releases a dioxygen (O) molecule as a waste product.
Sentence 1: Although photosynthesis is performed differently by different species , the process always begins when energy from light is absorbed by proteins called reaction centres that contain green chlorophyll pigments . In plants , these proteins are held inside organelles called chloroplasts , which are most abundant in leaf cells , while in bacteria they are embedded in the plasma membrane . In these light - dependent reactions , some energy is used to strip electrons from suitable substances , such as water , producing oxygen gas . The hydrogen freed by the splitting of water is used in the creation of two further compounds that act as an immediate energy storage means : reduced nicotinamide adenine dinucleotide phosphate ( NADPH ) and adenosine triphosphate ( ATP ) , the `` energy currency '' of cells .
  Fact 1.0: Photosynthesis is performed differently by different species.
  Fact 1.1: Photosynthesis always begins when energy from light is absorbed by proteins called reaction centres.
  Fact 1.2: Reaction centres are proteins that contain green chlorophyll pigments.
  Fact 1.3: In plants, reaction centres are held inside organelles called chloroplasts.
  Fact 1.4: Chloroplasts are most abundant in leaf cells.
  Fact 1.5: In bacteria, reaction centres are embedded in the plasma membrane.
  Fact 1.6: In the light-dependent reactions, some energy is used to strip electrons from suitable substances, such as water.
  Fact 1.7: The light-dependent reactions produce oxygen gas.
  Fact 1.8: The hydrogen freed by splitting water is used in the creation of two further compounds.
  Fact 1.9: The two compounds act as an immediate energy storage means.
  Fact 1.10: The two compounds are reduced nicotinamide adenine dinucleotide phosphate (NADPH) and adenosine triphosphate (ATP).
  Fact 1.11: ATP is the energy currency of cells.
Sentence 2: Photosynthesis occurs in two stages . In the first stage , light - dependent reactions or light reactions capture the energy of light and use it to make the energy - storage molecules ATP and NADPH . During the second stage , the light - independent reactions use these products to capture and reduce carbon dioxide .
  Fact 2.0: Photosynthesis occurs in two stages.
  Fact 2.1: In the first stage, light-dependent reactions (light reactions) capture the energy of light.
  Fact 2.2: In the first stage, the light-dependent reactions use the captured energy to make the energy-storage molecules ATP and NADPH.
  Fact 2.3: During the second stage, the light-independent reactions use ATP and NADPH.
  Fact 2.4: In the second stage, the light-independent reactions capture and reduce carbon dioxide.

Retrieved items:
<item 1>
Photosynthesis
but canceling n water molecules from each side gives the net equation : CO + H O + photons → ( CH O ) + O carbon dioxide + water + light energy → carbohydrate +
oxygen Other processes substitute other compounds ( such as arsenite ) for water in the electron - supply role ; for example some microbes use sunlight to
oxidize arsenite to arsenate : The equation for this reaction is : CO + ( AsO 3 − 3 ) + photons → ( AsO 3 − 4 ) + CO carbon dioxide + arsenite + light energy →
arsenate + carbon monoxide ( used to build other compounds in subsequent reactions ) Photosynthesis occurs in two stages . In the first stage , light -
dependent reactions or light reactions capture the energy of light and use it to make the energy - storage molecules ATP and NADPH . During the second stage ,
the light - independent reactions use these products to capture and reduce carbon dioxide . Most organisms that utilize oxygenic photosynthesis use visible
light for the light - dependent reactions , although at least three use shortwave infrared or , more specifically , far - red radiation . Some organisms employ
even more radical variants of photosynthesis .
</item 1>

<item 2>
Photosynthesis
The overall equation for the light - independent reactions in green plants is 3 CO + 9 ATP + 6 NADPH + 6 H → C H O - phosphate + 9 ADP + 8 P + 6 NADP + 3 H O
Overview of the Calvin cycle and carbon fixation Carbon fixation produces the intermediate three - carbon sugar product , which is then converted to the final
carbohydrate products . The simple carbon sugars produced by photosynthesis are then used in the forming of other organic compounds , such as the building
material cellulose , the precursors for lipid and amino acid biosynthesis , or as a fuel in cellular respiration . The latter occurs not only in plants but also
in animals when the energy from plants is passed through a food chain . The fixation or reduction of carbon dioxide is a process in which carbon dioxide
combines with a five - carbon sugar , ribulose 1 , 5 - bisphosphate , to yield two molecules of a three - carbon compound , glycerate 3 - phosphate , also known
as 3 - phosphoglycerate .
</item 2>

<item 3>
Photosynthesis
Louis N.M. Duysens and Jan Amesz discovered that chlorophyll a will absorb one light , oxidize cytochrome f , chlorophyll a ( and other pigments ) will absorb
another light , but will reduce this same oxidized cytochrome , stating the two light reactions are in series . Development of the concept In 1893 , Charles
Reid Barnes proposed two terms , photosyntax and photosynthesis , for the biological process of synthesis of complex carbon compounds out of carbonic acid , in
the presence of chlorophyll , under the influence of light . Over time , the term photosynthesis came into common usage as the term of choice . Later discovery
of anoxygenic photosynthetic bacteria and photophosphorylation necessitated redefinition of the term . C3 : C4 photosynthesis Research After WWII at late 1940
at the University of California , Berkeley , the details of photosynthetic carbon metabolism were sorted out by the chemists Melvin Calvin , Andrew Benson ,
James Bassham and a score of students and researchers utilizing the carbon - 14 isotope and paper chromatography techniques .
</item 3>

<item 4>
Photosynthesis
The source of electrons in green - plant and cyanobacterial photosynthesis is water . Two water molecules are oxidized by four successive charge - separation
reactions by photosystem II to yield a molecule of diatomic oxygen and four hydrogen ions ; the electrons yielded are transferred to a redox - active tyrosine
residue that then reduces the oxidized chlorophyll a ( called P680 ) that serves as the primary light - driven electron donor in the photosystem II reaction
center . That photo receptor is in effect reset and is then able to repeat the absorption of another photon and the release of another photo - dissociated
electron . The oxidation of water is catalyzed in photosystem II by a redox - active structure that contains four manganese ions and a calcium ion ; this oxygen
- evolving complex binds two water molecules and contains the four oxidizing equivalents that are used to drive the water - oxidizing reaction ( Dolai 's S -
state diagrams ) . Ref . Dolai , U ( 2017 ) . `` Chemical Scheme of Water - Splitting Process during Photosynthesis by the way of Experimental Analysis `` .
IOSR Journal of Pharmacy and Biological Sciences .
</item 4>

<item 5>
Photosynthesis
Light - dependent reactions Light - dependent reactions of photosynthesis at the thylakoid membrane Main article : Light - dependent reactions In the light -
dependent reactions ( mainly three vital reaction steps : ( I ) Hill and Bendall 's Z - scheme of cyclic and non-cyclic reactions , ( II ) Mitchell 's chemi -
osmosis for ATP synthesis and ( III ) Dolai 's S - state diagrams in water - splitting ) , one molecule of the pigment chlorophyll absorbs one photon and loses
one electron . This electron is passed to a modified form of chlorophyll called pheophytin , which passes the electron to a quinone molecule , starting the flow
of electrons down an electron transport chain that leads to the ultimate reduction of NADP to NADPH . In addition , this creates a proton gradient ( energy
gradient ) across the chloroplast membrane , which is used by ATP synthase in the synthesis of ATP . The chlorophyll molecule ultimately regains the electron it
lost when a water molecule is split in a process called photolysis , which releases a dioxygen ( O ) molecule as a waste product .
</item 5>

<item 6>
Photosynthesis
Z scheme The `` Z scheme '' In plants , light - dependent reactions occur in the thylakoid membranes of the chloroplasts where they drive the synthesis of ATP
and NADPH . The light - dependent reactions are of two forms : cyclic and non-cyclic . In the non-cyclic reaction , the photons are captured in the light -
harvesting antenna complexes of photosystem II by chlorophyll and other accessory pigments ( see diagram at right ) . The absorption of a photon by the antenna
complex frees an electron by a process called photoinduced charge separation . The antenna system is at the core of the chlorophyll molecule of the photosystem
II reaction center . That freed electron is transferred to the primary electron - acceptor molecule , pheophytin . As the electrons are shuttled through an
electron transport chain ( the so - called Z - scheme shown in the diagram ) , it initially functions to generate a chemiosmotic potential by pumping proton
cations ( H ) across the membrane and into the thylakoid space .
</item 6>

<item 7>
Photosynthesis
The cyclic reaction takes place only at photosystem I. Once the electron is displaced from the photosystem , the electron is passed down the electron acceptor
molecules and returns to photosystem I , from where it was emitted , hence the name cyclic reaction . Water photolysis Main articles : Photodissociation and
Oxygen evolution The NADPH is the main reducing agent produced by chloroplasts , which then goes on to provide a source of energetic electrons in other cellular
reactions . Its production leaves chlorophyll in photosystem I with a deficit of electrons ( chlorophyll has been oxidized ) , which must be balanced by some
other reducing agent that will supply the missing electron . The excited electrons lost from chlorophyll from photosystem I are supplied from the electron
transport chain by plastocyanin . However , since photosystem II is the first step of the Z - scheme , an external source of electrons is required to reduce its
oxidized chlorophyll a molecules . The source of electrons in green - plant and cyanobacterial photosynthesis is water .
</item 7>

<item 8>
Photosynthesis
IOSR Journal of Pharmacy and Biological Sciences . 12 ( 6 ) : 65 - 67 . doi : 10.9790 / 3008 - 1206026567 . ISSN 2319 - 7676 . Photosystem II is the only known
biological enzyme that carries out this oxidation of water . The hydrogen ions released contribute to the transmembrane chemiosmotic potential that leads to ATP
synthesis . Oxygen is a waste product of light - dependent reactions , but the majority of organisms on Earth use oxygen for cellular respiration , including
photosynthetic organisms . Light - independent reactions Calvin cycle Main articles : Calvin cycle , Carbon fixation , and Light - independent reactions In the
light - independent ( or `` dark '' ) reactions , the enzyme RuBisCO captures CO from the atmosphere and , in a process called the Calvin - Benson cycle , it
uses the newly formed NADPH and releases three - carbon sugars , which are later combined to form sucrose and starch .
</item 8>

<item 9>
Photosynthesis
A highly simplified summary is : 2 glycolate + ATP → 3 - phosphoglycerate + carbon dioxide + ADP + NH The salvaging pathway for the products of RuBisCO
oxygenase activity is more commonly known as photorespiration , since it is characterized by light - dependent oxygen consumption and the release of carbon
dioxide . See also Environment portal Ecology portal Earth sciences portal Metabolism portal Jan Anderson ( scientist ) Artificial photosynthesis Calvin -
Benson cycle Carbon fixation Cellular respiration Chemosynthesis Integrated fluorometer Light - dependent reaction Organic reaction Photobiology Photoinhibition
Photosynthetic reaction center Photosynthetically active radiation Photosystem Photosystem I Photosystem II Quantum biology Radiosynthesis Red edge Vitamin D
Hill reaction References Jump up ^ `` photosynthesis '' . Online Etymology Dictionary . Jump up ^ φῶς . Liddell , Henry George ; Scott , Robert ; A Greek --
English Lexicon at the Perseus Project Jump up ^ σύνθεσις .
</item 9>

<item 10>
Photosynthesis
The process is most productive when the absorption of quanta are equal in both the PSII and PSI , assuring that input energy from the antenna complex is divided
between the PSI and PSII system , which in turn powers the photochemistry . Melvin Calvin works in his photosynthesis laboratory . Robert Hill thought that a
complex of reactions consisting of an intermediate to cytochrome b ( now a plastoquinone ) , another is from cytochrome f to a step in the carbohydrate -
generating mechanisms . These are linked by plastoquinone , which does require energy to reduce cytochrome f for it is a sufficient reductant . Further
experiments to prove that the oxygen developed during the photosynthesis of green plants came from water , were performed by Hill in 1937 and 1939 . He showed
that isolated chloroplasts give off oxygen in the presence of unnatural reducing agents like iron oxalate , ferricyanide or benzoquinone after exposure to light
. The Hill reaction is as follows : 2 H O + 2 A + ( light , chloroplasts ) → 2 AH + O where A is the electron acceptor .
</item 10>
=== END ITEM id=cf763d122ba9d969 ===

=== ITEM id=e1fdd62c2eb67198 ===
Gold sentences and facts:
Sentence 0: Symphony No. 40 in G minor , KV. 550 was written by Wolfgang Amadeus Mozart in 1788 . It is sometimes referred to as the `` Great G minor symphony '' , to distinguish it from the `` Little G minor symphony '' , No. 25 . The two are the only extant minor key symphonies Mozart wrote .
  Fact 0.0: Symphony No. 40 in G minor, KV. 550 was written by Wolfgang Amadeus Mozart.
  Fact 0.1: Symphony No. 40 in G minor, KV. 550 was written in 1788.
  Fact 0.2: Symphony No. 40 in G minor, KV. 550 is sometimes referred to as the 'Great G minor symphony'.
  Fact 0.3: The Great G minor symphony is distinguished from the 'Little G minor symphony', No. 25.
  Fact 0.4: Symphony No. 40 and the Little G minor symphony, No. 25, are the only extant minor key symphonies Mozart wrote.

Retrieved items:
<item 1>
Symphony No. 40 (Mozart)
504 , Prague ) No. 39 ( K. 543 ) No. 40 ( K. 550 ) No. 41 ( K. 551 , Jupiter ) Lost K. 19b K. 66c K. 66d K. 66e K. Anh. C 11.07 K. Anh. C 11.08 No. 2 now
attributed to Leopold Mozart . No. 3 now attributed to Carl Friedrich Abel . No. 37 now attributed to Michael Haydn . List of symphonies by Wolfgang Amadeus
Mozart List of spurious / doubtful Mozart symphonies LCCN : n81018421 GND : 300111150 BNF : cb13915276b ( data ) Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Symphony_No._40_(Mozart)&oldid=825476175 '' Categories : Symphonies by Wolfgang Amadeus Mozart 1788 compositions
Compositions in G minor Hidden categories : All articles with specifically marked weasel - worded phrases Articles with specifically marked weasel - worded
phrases from
</item 1>

<item 2>
Symphony No. 40 (Mozart)
Franz Schubert likewise copied down the music of Mozart 's minuet , and the minuet of his Fifth Symphony strongly evokes Mozart 's . Zaslaw has suggested that a
passage late in Joseph Haydn 's oratorio The Seasons ( 1801 ) , a meditation on death , quotes the second movement of the 40th Symphony and was included by
Haydn as a memorial to his long - dead friend . First recording ( edit ) The first known recording of the 40th Symphony is by the Victor Recording Company and
issued in 1915 under the title `` Symphony in G Minor '' . The Victor Concert Orchestra performed under the direction of conductor Walter B. Rogers . Media (
edit ) The following files contain a digital recording of a performance of the 40th Symphony by the Fulda Symphonic Orchestra . The performance took place on
March 18 , 2001 in the Orangerie in Fulda , Germany . 1st movement Molto allegro 2nd movement Andante 3rd movement Menuetto , Allegretto - Trio 4th movement
Allegro assai Problems playing these files ? See media help .
</item 2>

<item 3>
Symphony No. 40 (Mozart)
Some commentators go further , suggesting that Mozart wrote the symphony ( and its companions , Nos. 39 and 41 ) without even intending it to be performed , but
rather for posterity ; as ( to use Alfred Einstein 's words ) , an `` appeal to eternity '' . Modern scholarship suggests that these conjectures are not correct
. First , in a recently discovered 10 July 1802 letter by the musician Johann Wenzel ( 1762 -- 1831 ) to the publisher Ambrosius Kühnel in Leipzig , Wenzel
refers to a performance of the symphony at the home of Baron Gottfried van Swieten with Mozart present , but the execution was so poor that the composer had to
leave the room . There is strong circumstantial evidence for other , probably better , performances . On several occasions between the composition of the
symphony and the composer 's death , symphony concerts were given featuring Mozart 's music for which copies of the program have survived , announcing a
symphony unidentified by date or key .
</item 3>

<item 4>
Symphony No. 40 (Mozart)
Zaslaw , Neal , and William Cowdery ( 1990 ) . The Compleat Mozart : A Guide to the Musical Works of Wolfgang Amadeus Mozart . New York : W.W. Norton & Company
. Extracts cited are viewable online at Google Books : ( 6 ) . External links ( edit ) Wikimedia Commons has media related to Symphony No. 40 ( Mozart ) .
Versions of the score : Sinfonie in g minor K. 550 ( 1st version ) : Score and critical report ( in German ) in the Neue Mozart - Ausgabe Sinfonie in g minor K.
550 ( 2nd version ) : Score in the Neue Mozart - Ausgabe Symphony No. 40 : Scores at the International Music Score Library Project ( IMSLP ) Kalmus miniature
editions , from the William and Gayle Cook Music Library at the Indiana University Jacobs School of Music A Public Domain version by the Columbia University
Orchestra at TEOC Classical Music Radio : Symphony No.
</item 4>

<item 5>
Symphony No. 40 (Mozart)
Nikolaus Harnoncourt conjectured that Mozart composed the three symphonies as a unified work , pointing , among other things , to the fact that the Symphony No.
40 , as the middle work , has no introduction ( unlike No. 39 ) and does not have a finale of the scale of No. 41 's . The 40th symphony exists in two versions
, differing primarily in that one includes parts for a pair of clarinets ( with suitable adjustments made in the other wind parts ) . Most likely , the clarinet
parts were added in a revised version . The autograph scores of both versions were acquired in the 1860s by Johannes Brahms , who later donated the manuscripts
to the Gesellschaft der Musikfreunde in Vienna , where they reside today . Premiere ( edit ) As Neal Zaslaw has pointed out , writers on Mozart have often
suggested -- or even asserted -- that Mozart never heard his 40th Symphony performed . Some commentators go further , suggesting that Mozart wrote the symphony
( and its companions , Nos.
</item 5>

<item 6>
Symphony No. 40 (Mozart)
Passage quoted may be viewed on line at Google Books : ( 5 ) . Schoenberg , Arnold ( 1954 , rvs. 1969 ) . Structural Functions of Harmony . New York : W.W.
Norton & Company . Analyzes the wide - ranging development sections of both outer movements at some length . Swafford , Jan ( 1997 ) . Johannes Brahms : A
Biography . New York : Alfred A. Knopf . Zaslaw , Neal ( 1983 ) . Introductory notes to a recording of the 31st and 40th Symphonies made by Christopher Hogwood
and the Academy of Ancient Music , Oiseau - Lyre 410 - 197 - 2 . Zaslaw , Neal ( 1994 ) . `` Mozart as Working Stiff '' . In James M. Morris , ed . On Mozart ,
. Washington : Woodrow Wilson Center Press . Zaslaw , Neal , and William Cowdery ( 1990 ) .
</item 6>

<item 7>
Mozart completed Symphony No. 40 in G minor, K. 550, on July 25, 1788, during the few weeks in which he also wrote Symphonies 39 and 41. One of his only two
surviving minor-key symphonies, it exists in versions with and without clarinets; the revision suggests Mozart expected performances, despite uncertainty about
its premiere. Its four movements last about 25 minutes and feature a restless opening, lyrical Andante,
</item 7>

<item 8>
Symphony No. 40 (Mozart)
Jump up ^ Milada Jonášová : `` Eine Aufführung der g - moll - Sinfonie KV 550 bei Baron van Swieten im Beisein Mozarts '' , in : Mozart Studien 20 , Tutzing
2011 , pp. 253 -- 268 . An abridged English translation was published in the Newsletter of the Mozart Society of America 16.1 ( 2012 ) , currently available on
line at ( 2 ) . The letter reads `` im Wien habe ich selbst es von verstorbenem Mozart gehört , als Er sie bei Baron Wanswiten ( sic ) hat produciren lassen ,
das ( s ) er wärend der production aus dem Zimmer sich hat entfernen müssen , wie man Sie unrichtig aufgeführt hat '' , meaning `` and in Vienna I have heard
myself from the departed Mozart , that when he had it performed in Baron Wanswiten 's rooms , he had to leave the room during the performance because it was
being played so incorrectly '' . Jump up ^ List from Zaslaw 1983 .
</item 8>

<item 9>
Symphony No. 40 (Mozart)
The work is in four movements , in the usual arrangement for a classical - style symphony ( fast movement , slow movement , minuet , fast movement ) : Molto
allegro , Andante , Menuetto . Allegretto -- Trio , Finale . Allegro assai , The first movement begins darkly , not with its first theme but with accompaniment
, played by the lower strings with divided violas . The technique of beginning a work with an accompaniment figure was later used by Mozart in his last piano
concerto ( KV. 595 ) and later became a favourite of the Romantics ( examples include the openings of Mendelssohn 's Violin Concerto and Sergei Rachmaninoff 's
Third Piano Concerto ) . The first theme is as follows . The second movement is a lyrical work in time . It is in the subdominant key of the relative major of G
minor ( B ♭ major ) : E ♭ major .
</item 9>

<item 10>
Symphony No. 40 (Mozart)
Symphony No. 40 ( Mozart ) - wikipedia Symphony No. 40 ( Mozart ) Jump to : navigation , search Symphony No. 40 in G minor , KV. 550 was written by Wolfgang
Amadeus Mozart in 1788 . It is sometimes referred to as the `` Great G minor symphony '' , to distinguish it from the `` Little G minor symphony '' , No. 25 .
The two are the only extant minor key symphonies Mozart wrote . Contents ( hide ) 1 Composition 2 Premiere 3 The music 4 Reception 5 Influence 6 First recording
7 Media 8 Notes 9 References 10 External links Composition ( edit ) The date of completion of this symphony is known exactly , since Mozart in his mature years
kept a full catalog of his completed works ; he entered the 40th Symphony into it on 25 July 1788 . Work on the symphony occupied an exceptionally productive
period of just a few weeks during which time he also completed the 39th and 41st symphonies ( 26 June and 10 August , respectively ) .
</item 10>
=== END ITEM id=e1fdd62c2eb67198 ===

=== ITEM id=f905603077133dd6 ===
Gold sentences and facts:
Sentence 0: Subjects are all taken on the same level - there is no higher or standard grade as in the past . There are three pass levels , each with different pass requirements. First is a pass with Higher certificate , one must obtain 40 % in home language , 40 % in two other subjects and must obtain 30 % in three other subjects . A learner who passed matric with higher certificate level , can not enroll for a University Degree nor a Diploma at an institute of higher learning . The mean mark in any subject is usually about 55 . Only a small proportion of candidates score an ' A ' in any subject ( from as little as 2 % to a maximum of about 10 % in subjects taken by highly select groups . ) A further 8 -- 15 % are likely to gain a ' B ' and about 20 -- 25 % achieve a ' C ' grade . The National Senior Certificate is a group certificate and records an aggregate mark .
  Fact 0.0: All subjects are taken on the same level, and there is no higher or standard grade as in the past.
  Fact 0.1: There are three pass levels, each with different pass requirements.
  Fact 0.2: The first pass level is a pass with Higher certificate.
  Fact 0.3: To pass with Higher certificate, a learner must obtain 40% in home language, 40% in two other subjects, and 30% in three other subjects.
  Fact 0.4: A learner who passed matric with Higher certificate level cannot enroll for a University Degree.
  Fact 0.5: A learner who passed matric with Higher certificate level cannot enroll for a Diploma at an institute of higher learning.
  Fact 0.6: The mean mark in any subject is usually about 55.
  Fact 0.7: Only a small proportion of candidates score an 'A' in any subject, from as little as 2% to a maximum of about 10% in subjects taken by highly select groups.
  Fact 0.8: A further 8 to 15% of candidates are likely to gain a 'B' in a subject.
  Fact 0.9: About 20 to 25% of candidates achieve a 'C' grade.
  Fact 0.10: The National Senior Certificate is a group certificate.
  Fact 0.11: The National Senior Certificate records an aggregate mark.

Retrieved items:
<item 1>
National Senior Certificate
National Senior certificate - wikipedia National Senior certificate Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( April 2010 ) ( Learn how and
when to remove this template message ) The National Senior Certificate or NSC is a high school diploma and is the main school - leaving certificate in South
Africa . This certificate is commonly known as the matriculation ( matric ) certificate , as grade 12 is the matriculation grade . The NSC , previously known as
the Further Education and Training Certificate or FETC , replaced the Senior Certificate with effect from 2008 and was phased in starting with grade 10 in 2006
. Pupils study at least seven subjects , including two compulsory official South African languages , either Mathematics or Mathematical Literacy , Life
Orientation and three elective subjects . Subjects are all taken on the same level - there is no higher or standard grade as in the past . There are three pass
levels , each with different pass requirements.
</item 1>

<item 2>
National Senior Certificate
Citizenship and Responsibility , this section teaches learners about their rights and their duties as citizens . Assessment ( edit ) There are three types of
subjects : General Subjects An end - of - year examination : 75 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Practical
Subjects An end - of - year examination : 50 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Practical component : 25 % of the
total mark Language Subjects An end - of - year examination : 50 % of the total mark Portfolio ( Continuous Assessment ) : 25 % of the total mark Oral
Examination : 25 % of the total mark Continuous Assessment ( CASS ) includes all the tests , examinations , tasks , activities , orals and projects done
throughout the year . Matric results are usually out of 400 marks .
</item 2>

<item 3>
National Senior Certificate
The IEB operates on a national level catering primarily for independent schools . Contents ( hide ) 1 Subject requirements 1.1 Group A : Fundamentals (
Compulsory ) 1.2 Group B : Electives 2 Life Orientation 3 Assessment 4 Level system ( 1 to 7 ) 5 The 5 % language compensation 6 University entrance 7 See also
8 References 9 External links Subject requirements. ( edit ) Learners study at least 7 subjects - 4 compulsory and at least 3 electives . All subjects are
written on one grade only and are no longer written on Higher or Standard Grade . Not all schools offer the full range of Elective subjects listed here . Each
school may offer subjects specific to its academic orientation . For example , Agriculture Schools offer the agriculture - orientated subjects whereas technical
Schools offer the practical and mechanical - orientated subjects.
</item 3>

<item 4>
National Senior Certificate
( Written reply to parliamentary question in 2011 ) `` The compensation applies to learners whose first language is neither English nor Afrikaans and who offer
an African language as their first language . They receive an additional 5 % on their non-language subjects . The measure was first introduced in 1999 by the
South African Certification Council . '' ( Written reply to parliamentary question in 2007 , ) Dr Sizwe Mabizela , Chairperson of Council , Umalusi has provided
further explanation : `` This is the most misunderstood concept in this country . In terms of the policy on language compensation , learners who offer an
African language as Home Language and do not offer Afrikaans or English as Home language qualify for a 5 % language compensation on the mark they have obtained
in a non-language subject .
</item 4>

<item 5>
National Senior Certificate
For example a learner who obtains a mark of zero ( 0 ) out of 300 will obtain 5 % of zero ( which is zero ) for language compensation ; a learner who obtains 10
out of 300 will receive 5 % of 10 , which is 0 , 5 marks , for language compensation ; a learner who obtains 100 out of 300 marks will obtain an additional 5
marks for language compensation . '' This kind of compensation has some impact on pass rates , but does provide significant impact at the upper end of the scale
, affecting those applying for admission to university . For example , a qualifying learner obtaining 95 % would receive 95 x 1.05 = 99.75 % ( which rounds to
100 % ) . A learner obtaining 40 % would receive 40 x 1.05 = 42.0 % . University entrance ( edit ) In addition to minimum grades required in each subject ,
universities either set their own entrance tests and / or use the National Benchmark Tests ( NBT ) .
</item 5>

<item 6>
National Senior Certificate
To study for a bachelor 's degree at a South African university requires that the applicant has at least an NSC endorsed by Umalusi , with a pass of 30 % in the
chosen university 's language of learning and teaching , as well as a level 4 or higher in the following list of designated , 19 - credit subjects : Accounting
Agricultural Sciences Business Studies Dramatic Arts Economics Engineering Graphics and Design Geography History Consumer Studies Information Technology
Languages ( one language of learning and teaching at a higher education institution and two other recognised language subjects ) Life Sciences Mathematics
Mathematical Literacy Music Physical Sciences Religion Studies Visual Arts See also ( edit ) High school # South Africa Matriculation in South Africa
Independent Examinations Board References ( edit ) Jump up ^ SOUTH AFRICAN NATIONAL ASSEMBLY : QUESTION 757 , DATE OF PUBLICATION OF INTERNAL QUESTION PAPER :
11 / 03 / 2011 ( INTERNAL QUESTION PAPER : 06 / 2011 ) Jump up ^ SOUTH AFRICAN NATIONAL ASSEMBLY : WRITTEN REPLY TO QUESTION 469 , DATE OF PUBLICATION IN
INTERNAL QUESTION PAPER : 26 - 03 - 2007 ( INTERNAL QUESTION PAPER NO.
</item 6>

<item 7>
National Senior Certificate
: 10 - 2007 ) Jump up ^ Politicsweb , 3 January 2011 , `` Umalusi explains matric mark adjustments '' , Dr Sizwe Mabizela , 03 January 2011 Jump up ^
http://nbt.ac.za/ About NBT Retrieved 21 March 2011 Jump up ^ http://wced.pgwc.gov.za/circulars/circulars08/e33_08.html External links ( edit ) Department of
Education official site IEB official site Retrieved from `` https://en.wikipedia.org/w/index.php?title=National_Senior_Certificate&oldid=837110357 '' Categories
: Education in South Africa Secondary school qualifications Hidden categories : Articles needing additional references from April 2010 All articles needing
additional references Talk Contents About Wikipedia Add links This page was last edited on 18 April 2018 , at 19 : 39 . About Wikipedia
</item 7>

<item 8>
National Senior Certificate
Matric results are usually out of 400 marks . Level system ( 1 to 7 ) ( edit ) Level 7 : 80 - 100 % ( Outstanding achievement ) Level 6 : 70 - 79 % (
Meritorious achievement ) Level 5 : 60 - 69 % ( Substantial achievement ) Level 4 : 50 - 59 % ( Moderate achievement ) Level 3 : 40 - 49 % ( Adequate
achievement ) Level 2 : 30 - 39 % ( Elementary achievement ) Level 1 : 0 - 29 % ( Not achieved - Fail ) The 5 % language compensation ( edit ) Language
compensation is described by several sources : `` To the final mark is added the language compensation , which is 5 % of the mark attained by the candidate for
all non-language subjects , for candidates whose mother tongue is not English or Afrikaans . The 5 % compensates learners for the disadvantage suffered by these
candidates being instructed in a language that is not their mother tongue . ''
</item 8>

<item 9>
National Senior Certificate
V Group a : fundamentals ( compulsory ) ( edit ) Learners must study 4 compulsory subjects , Two official languages of South Africa must be studied : Afrikaans
English Ndebele Northern Sotho Sesotho Siswati Tsonga Tswana Venda Xhosa Zulu Mathematics or Mathematical Literacy Life Orientation Group B : electives ( edit )
At least 3 subjects from the following : Accounting Agricultural Management Practices Agricultural Sciences Agricultural Technology Business Studies Civil
Technology Computer Applications Technology ( previously Computer Studies Standard Grade - Literacy ) Consumer Studies Dance Studies Design Dramatic Arts
Economics Electrical Technology Engineering Graphics & Design Geography History Hospitality Studies ( previously Home Economics ) Information Technology (
previously Computer Studies Higher Grade - Programming ) Life Sciences ( previously Biology ) Mechanical Technology Music Physical Science Religion Studies
Second Additional Language Third Additional Language Tourism Visual Arts Life Orientation ( edit ) Life Orientation ( colloquially abbreviated as `` LO '' ) has
been introduced into the senior high school phase as an examination subject . Life Orientation is a broad - learning subject that covers non-academic skills
needed in everyday life .
</item 9>

<item 10>
National Senior Certificate
There are three pass levels , each with different pass requirements. First is a pass with Higher certificate , one must obtain 40 % in home language , 40 % in
two other subjects and must obtain 30 % in three other subjects . A learner who passed matric with higher certificate level , can not enroll for a University
Degree nor a Diploma at an institute of higher learning . The mean mark in any subject is usually about 55 . Only a small proportion of candidates score an ' A
' in any subject ( from as little as 2 % to a maximum of about 10 % in subjects taken by highly select groups . ) A further 8 -- 15 % are likely to gain a ' B '
and about 20 -- 25 % achieve a ' C ' grade . The National Senior Certificate is a group certificate and records an aggregate mark . The Department of Basic
Education has responsibility for general educational policy to be implemented by nine provincial education departments and private providers such as the
Independent Examinations Board ( IEB ) . There are nine provincial examination boards and three independent boards , of which the IEB is the biggest . The IEB
operates on a national level catering primarily for independent schools .
</item 10>
=== END ITEM id=f905603077133dd6 ===

=== ITEM id=ff1cdf4d24a8bf27 ===
Gold sentences and facts:
Sentence 0: Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher Usmanov ( 30.04 % ) 15 ! $22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M $3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 ! $2.3 B Steel and energy Theatre production Planet Hollywood Ocean Software Huddersfield Town Dean Hoyle 09 ! $475 M Card Factory Leicester City The Srivaddhanaprabha Family 10 ! $4.9 B King Power International Group Liverpool ( more information ) John W. Henry 11 ! $2.5 B Fenway Sports Group Manchester City ( more information ) Mansour bin Zayed Al Nahyan ( 87 % ) China Media Capital ( 13 % ) 01 ! $44 B Oil and industry investment Government Manchester United ( more information ) The Glazer Family 06 ! $4.9 B First Allied Corporation , Tampa Bay Buccaneers Newcastle United ( more information ) Mike Ashley 04 ! $3.3 B Sports Direct International Southampton Gao Jisheng ( 80 % ) Katharina Liebherr ( 20 % ) 07 ! $3.9 B + Lander Sports Development Co Ltd Inheritance Stoke City The Coates Family 08 ! $3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B -- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport , Sunday Sport Gold Group International Straumur Investment Bank
  Fact 0.0: Arsenal's owner E. Stanley Kroenke holds 67.05% of the club.
  Fact 0.1: Arsenal's owner Alisher Usmanov holds 30.04% of the club.
  Fact 0.2: Arsenal's estimated combined owner net worth is $22.4 B.
  Fact 0.3: Arsenal's owners' source of wealth is metal and mining and investment.
  Fact 0.4: Kroenke Sports Enterprises is listed for Arsenal.
  Fact 0.5: Bournemouth's owner Maxim Demin holds 75% of the club.
  Fact 0.6: Bournemouth's owner PEAK6 Investments holds 25% of the club.
  Fact 0.7: Bournemouth's source of wealth is petrochemicals trading.
  Fact 0.8: Seven Developments is listed for Bournemouth.
  Fact 0.9: Brighton & Hove Albion's owner Tony Bloom holds 75.61% of the club.
  Fact 0.10: Brighton & Hove Albion's owner net worth is $1.7 B.
  Fact 0.11: Brighton & Hove Albion's source of wealth is online gambling, real estate, property, land development and investments.
  Fact 0.12: Burnley's owner Mike Garlick holds 49.3% of the club.
  Fact 0.13: Burnley's owner John Banaszkiewicz holds 27.55% of the club.
  Fact 0.14: Burnley's estimated combined owner net worth is $80 M.
  Fact 0.15: Burnley's sources of wealth are Michael Bailey Associates and Freight Investor Services.
  Fact 0.16: Chelsea's owner is Roman Abramovich.
  Fact 0.17: Chelsea's owner net worth is $9.3 B.
  Fact 0.18: Chelsea's source of wealth is oil and industry.
  Fact 0.19: Crystal Palace's owners are Steve Parish, Martin Long, Joshua Harris, David S. Blitzer, Jeremy Hosking and Stephen Browett.
  Fact 0.20: Crystal Palace's owners' net worths are listed as $65 M, $100 M, $3.2 B, $1.3 B, $250 M and $110 M.
  Fact 0.21: Crystal Palace's sources of wealth are listed as Tag Worldwide, Churchill Insurance, Apollo Global Management, Blackstone Group, Marathon Asset Management and Farr Vintners.
  Fact 0.22: Everton's owner Farhad Moshiri holds 49.9% of the club.
  Fact 0.23: Everton's owners include Bill Kenwright CBE and Jon Woods.
  Fact 0.24: Everton's estimated combined owner net worth is $2.3 B.
  Fact 0.25: Everton's sources of wealth are steel and energy, theatre production, Planet Hollywood and Ocean Software.
  Fact 0.26: Huddersfield Town's owner is Dean Hoyle.
  Fact 0.27: Huddersfield Town's owner net worth is $475 M.
  Fact 0.28: Huddersfield Town's source of wealth is Card Factory.
  Fact 0.29: Leicester City's owner is The Srivaddhanaprabha Family.
  Fact 0.30: Leicester City's owner net worth is $4.9 B.
  Fact 0.31: Leicester City's source of wealth is King Power International Group.
  Fact 0.32: Liverpool's owner is John W. Henry.
  Fact 0.33: Liverpool's owner net worth is $2.5 B.
  Fact 0.34: Liverpool's source of wealth is Fenway Sports Group.
  Fact 0.35: Manchester City's owner Mansour bin Zayed Al Nahyan holds 87% of the club.
  Fact 0.36: Manchester City's owner China Media Capital holds 13% of the club.
  Fact 0.37: Manchester City's owner net worth is $44 B.
  Fact 0.38: Manchester City's sources of wealth are oil and industry investment and government.
  Fact 0.39: Manchester United's owner is The Glazer Family.
  Fact 0.40: Manchester United's owner net worth is $4.9 B.
  Fact 0.41: Manchester United's sources of wealth are First Allied Corporation and Tampa Bay Buccaneers.
  Fact 0.42: Newcastle United's owner is Mike Ashley.
  Fact 0.43: Newcastle United's owner net worth is $3.3 B.
  Fact 0.44: Newcastle United's source of wealth is Sports Direct International.
  Fact 0.45: Southampton's owner Gao Jisheng holds 80% of the club.
  Fact 0.46: Southampton's owner Katharina Liebherr holds 20% of the club.
  Fact 0.47: Southampton's owner net worth is $3.9 B.
  Fact 0.48: Southampton's source of wealth is Lander Sports Development Co Ltd and inheritance.
  Fact 0.49: Stoke City's owner is The Coates Family.
  Fact 0.50: Stoke City's owner net worth is $3.1 B.
  Fact 0.51: Stoke City's source of wealth is Bet365.
  Fact 0.52: Swansea City's owners Stephen Kaplan and Jason Levien hold 68% of the club.
  Fact 0.53: Swansea City's Supporters Trust holds 21.1% of the club.
  Fact 0.54: Swansea City's sources of wealth are Memphis Grizzlies, D.C. United and investment trust.
  Fact 0.55: Tottenham Hotspur's owner Joe Lewis holds 70.6% of the 85% holding company.
  Fact 0.56: Tottenham Hotspur's owner Daniel Levy holds 29.4% of the 85% holding company.
  Fact 0.57: Tottenham Hotspur's owner net worth is $5.6 B.
  Fact 0.58: Tottenham Hotspur's sources of wealth are Tavistock Group and ENIC Group.
  Fact 0.59: Watford's owner is Gino Pozzo.
  Fact 0.60: Watford's owner is described as a businessman.
  Fact 0.61: West Bromwich Albion's owner Guochuan Lai holds 87.8% of the club.
  Fact 0.62: West Bromwich Albion's owner net worth is $6.2 B.
  Fact 0.63: West Bromwich Albion's source of wealth is investment.
  Fact 0.64: West Ham United's owner David Sullivan holds 51.1% of the club.
  Fact 0.65: West Ham United's owner David Gold holds 35.1% of the club.
  Fact 0.66: West Ham United's owner Straumur Investment Bank holds 10% of the club.
  Fact 0.67: West Ham United's owners' net worths are listed as $520 M and $325 M.
  Fact 0.68: West Ham United's sources of wealth are Daily Sport, Sunday Sport, Gold Group International and Straumur Investment Bank.
Sentence 1: Club Owner ( s ) Estimated combined net worth ( £ m ) Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % ) Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % ) Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company Middlesbrough Steve Gibson 10 ! $260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn - Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C. Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 ! $155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment
  Fact 1.0: Aston Villa's owner is Dr. Tony Xia.
  Fact 1.1: Aston Villa's owner net worth is $1.2 B.
  Fact 1.2: Aston Villa's source of wealth is Recon Group.
  Fact 1.3: Barnsley's owner is Patrick Cryne.
  Fact 1.4: Barnsley's owner net worth is $40 M.
  Fact 1.5: Barnsley's source of wealth is iSOFT.
  Fact 1.6: Birmingham City's owners are Paul Suen (60.78%), Wang Lei (5.28%) and Carson Yeung (6.51%).
  Fact 1.7: Birmingham City's estimated combined owner net worth is $570 M.
  Fact 1.8: Birmingham City's source of wealth is Birmingham International Holdings.
  Fact 1.9: Bolton Wanderers' owner is Eddie Davies CBE.
  Fact 1.10: Bolton Wanderers' owner net worth is $52 M.
  Fact 1.11: Bolton Wanderers' source of wealth is Strix.
  Fact 1.12: Brentford's owner Matthew Benham holds 96% of the club.
  Fact 1.13: Bristol City's owner is Stephen Lansdown.
  Fact 1.14: Bristol City's owner net worth is $1.9 B.
  Fact 1.15: Bristol City's source of wealth is Hargreaves Lansdown.
  Fact 1.16: Burton Albion's owner is Ben Robinson.
  Fact 1.17: Cardiff City's owner Vincent Tan holds 51% of the club.
  Fact 1.18: Cardiff City's owner net worth is $820 M.
  Fact 1.19: Cardiff City's source of wealth is Berjaya Group.
  Fact 1.20: Derby County's owner is Mel Morris.
  Fact 1.21: Derby County's owner net worth is $520 M.
  Fact 1.22: Fulham's owner is Shahid Khan.
  Fact 1.23: Fulham's owner net worth is $8.9 B.
  Fact 1.24: Fulham's source of wealth is Flex-N-Gate Corp, an automobile parts manufacturer.
  Fact 1.25: Hull City's owner is The Allam Family.
  Fact 1.26: Hull City's owner net worth is $325 M.
  Fact 1.27: Hull City's source of wealth is Allam Marine.
  Fact 1.28: Ipswich Town's owner is Marcus Evans.
  Fact 1.29: Ipswich Town's owner net worth is $825 M.
  Fact 1.30: Ipswich Town's source of wealth is Marcus Evans Group.
  Fact 1.31: Leeds United's owners are Andrea Radrizzani, Eleonora Sports Ltd and Aser Group Company.
  Fact 1.32: Middlesbrough's owner is Steve Gibson.
  Fact 1.33: Middlesbrough's owner net worth is $260 M.
  Fact 1.34: Middlesbrough's source of wealth is Bulkhaul Limited.
  Fact 1.35: Millwall's owners are John Berylson and Richard Smith.
  Fact 1.36: Millwall's owner net worth is $175 M.
  Fact 1.37: Millwall's source of wealth is Chestnut Hill Ventures.
  Fact 1.38: Norwich City's owners Delia Smith and Michael Wynn-Jones hold 53% of the club.
  Fact 1.39: Norwich City's owner Michael Foulger holds 15% of the club.
  Fact 1.40: Norwich City's owner net worth is $30 M.
  Fact 1.41: Norwich City's sources of wealth are food industry, publishing and poultry.
  Fact 1.42: Nottingham Forest's owner is Evangelos Marinakis.
  Fact 1.43: Nottingham Forest's owner net worth is $650 M.
  Fact 1.44: Nottingham Forest's owner is also owner of Olympiacos F.C.
  Fact 1.45: Preston North End's owner Trevor Hemmings CVO holds 51.4% of the club.
  Fact 1.46: Preston North End's owner net worth is $720 M.
  Fact 1.47: Preston North End's sources of wealth are Blackpool Tower, horse racing, investment, Northern Trust and Sportech.
  Fact 1.48: Queens Park Rangers' owner Tony Fernandes holds 66% of the club.
  Fact 1.49: Queens Park Rangers' owner Lakshmi Mittal holds 33% of the club.
  Fact 1.50: Queens Park Rangers' owners' net worths are listed as $345 M and $16.4 B.
  Fact 1.51: Queens Park Rangers' sources of wealth are Air Asia, Tune Group and ArcelorMittal.
  Fact 1.52: Reading's owners are Narin Niruttinanon (50%), Sasima Srivikorn (25%) and Sumrith Thanakarnjanasuth (25%).
  Fact 1.53: Sheffield United's owner Abdullah bin Musa'ed holds 50% of the club.
  Fact 1.54: Sheffield United's owner net worth is $155 M.
  Fact 1.55: Sheffield United's sources of wealth are property and oil.
  Fact 1.56: Sheffield Wednesday's owner is Dejphon Chansiri.
  Fact 1.57: Sheffield Wednesday's source of wealth is Thai Union Group.
  Fact 1.58: Sunderland's owner is Ellis Short.
  Fact 1.59: Sunderland's owner net worth is $2.9 B.
  Fact 1.60: Sunderland's sources of wealth are Lone Star Funds and Skibo Castle.
  Fact 1.61: Wolverhampton Wanderers' owners are Fosun International (parent company) and Guo Guangchang (co-owner of Fosun).
  Fact 1.62: Wolverhampton Wanderers' owner net worth is $6.2 B.
  Fact 1.63: Wolverhampton Wanderers' source of wealth is investment.

Retrieved items:
<item 1>
List of owners of English football clubs
$155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton
Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment League One ( edit ) Club Owner ( s ) Estimated combined
net worth Source of wealth AFC Wimbledon Wimbledon Football Club Supporters ' Society ( The Dons Trust ) Supporters Trust Blackburn Rovers Balaji & Venkatesh
Rao ( 99.9 % ) 04 ! $650 M Venkateshwara Hatcheries Group Bradford City Bristol Rovers Dwane Sports ( 92 % ) Banking and tourism Bury Save Our Shakers Trust (
63.8 % ) The Bury F.C. Supporters Society Ltd ( Forever Bury ) ( 11 % ) Investment trusts Charlton Athletic Roland Duchatelet ( 95 % ) 24 ! Staprix NV Doncaster
Rovers John Ryan Dick Watson ( 26 % ) Terry Bramall CBE 07 !
</item 1>

<item 2>
List of owners of English football clubs
List of owners of English football clubs - wikipedia List of owners of English football clubs Jump to : navigation , search This is a list of the current owners
of English football clubs , as well as ( in some cases ) their estimated net worth and source of wealth . All figures are listed in millions rather than
billions . For example , £ 1,000 m would be one billion ( 1,000,000,000 ) pounds . Only shareholders with a significant interest ( above 10 % ) are listed .
Contents ( hide ) 1 Premier League 2 Football League Championship 3 League One 4 League Two 5 References Premier League ( edit ) This list is incomplete ; you
can help by expanding it . Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher
Usmanov ( 30.04 % ) 15 !
</item 2>

<item 3>
List of owners of English football clubs
`` Trevor Hemmings ' Northern Trust returns to profit '' . The Daily Telegraph . London . Retrieved 12 February 2013 . Jump up ^ `` Company details '' . Queens
Park Rangers F.C. 10 July 2012 . Archived from the original on 10 August 2012 . Retrieved 12 February 2012 . Jump up ^ `` # 28 Anthony Fernandes '' . Forbes .
Retrieved 27 May 2014 . Jump up ^ `` # 56 Lakshmi Mittal '' . Forbes . Retrieved 19 May 2017 . Jump up ^ Details , Company . `` COMPANY DETAILS '' . Retrieved
29 July 2016 . Jump up ^ `` Fantasy football nut Saudi Prince Abdullah is joint owner of League One side Sheffield United '' . Daily Mail . London . 3 September
2013 . Retrieved 5 September 2013 . Jump up ^ `` Board & Management '' . A.F.C. Sunderland . Retrieved 9 May 2014 .
</item 3>

<item 4>
List of owners of English football clubs
$3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United
Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B
-- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment
West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport ,
Sunday Sport Gold Group International Straumur Investment Bank Football League Championship ( edit ) Club Owner ( s ) Estimated combined net worth ( £ m )
Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B !
</item 4>

<item 5>
List of owners of English football clubs
Jump up ^ `` Shahid Khan '' . Forbes . Forbes.com LLC. 19 May 2017 . Retrieved 19 May 2017 . Jump up ^ `` Who 's Who At The KC '' . Hull City F.C. 25 June 2012
. Retrieved 9 May 2014 . Jump up ^ `` Sunday Times Rich List 2017 reveals Yorkshire 's 20 wealthiest people '' . Hull Daily Mail . Archived from the original on
2 May 2017 . Retrieved 19 May 2017 . Jump up ^ www.calcioefinanza.it Leeds http://www.calcioefinanza.it/2017/01/04/leeds-cellino-radrizzani/ Leeds Check url =
value ( help ) . Retrieved 2017 - 01 - 10 . Missing or empty title = ( help ) Jump up ^ leeds united
https://www.leedsunited.com/news/club-news/21921/club-statement . Retrieved 2017 - 05 - 23 .
</item 5>

<item 6>
List of owners of English football clubs
$250 M DW Sports Fitness , JJB Sports League Two ( edit ) Club Owner ( s ) Estimated combined net worth Source of wealth Accrington Stanley The Styring Family (
12 % ) Accrington Stanley Supporters Fund ( 12 % ) Peter Marsden ( 10 % ) Investment trust Barnet Blackpool Owen Oyston ( 80 % ) Valeri Belokon ( 25 % ) 07 !
$105 M $195 M Publishing , Media & Estate Agency Finance , Media , Food Industry & Mining Cambridge United Grosvenor Fund Management Grosvenor Group Owned by
the Duke of Westminster and family Property Carlisle United ( more information ) A. Jenkins ( 37 % of 93 % holding company ) J. Nixon ( 18 % ) S. Pattison ( 18
% ) The United Trust ( 25.4 % ) Pioneer Foods Businessman ( retired ) Carlisle Glass Investment Trust Cheltenham Town Paul Baker Chesterfield Dave Allen
Colchester United Robbie Cowling 09 !
</item 6>

<item 7>
List of owners of English football clubs
^ Jump up to : `` The Football Rich List 2011 / 12 : The Full 100 '' . Four Four Two . Retrieved 13 February 2013 . Jump up ^ `` Board of Directors / CEO '' .
Burnley F.C . Retrieved 9 May 2014 . Jump up ^ `` The Rich List 2014 '' . Hertfordshire Life . 31 March 2014 . Retrieved 12 May 2014 . Jump up ^ `` Club
Personnel '' . Chelsea F.C . Retrieved 9 August 2015 . Jump up ^ `` # 139 Roman Abramovich '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` # 745 Joshua
Harris '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Senior changes at Tag Worldwide '' . Williams Lea. 24 October 2012 . Retrieved 16 February 2013 . Jump
up ^ `` # 896 Farhad Moshiri '' . Forbes . Retrieved 19 May 2017 .
</item 7>

<item 8>
List of owners of English football clubs
$260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn -
Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C.
Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more
information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin
Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 !
</item 8>

<item 9>
List of owners of English football clubs
Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % )
Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % )
Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby
County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam
Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company
Middlesbrough Steve Gibson 10 !
</item 9>

<item 10>
List of owners of English football clubs
Brentford F.C . Retrieved 29 May 2014 . Jump up ^ `` # 1098 Stephen Lansdown '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Club Ownership '' . Cardiff City
F.C . Retrieved 30 November 2013 . Jump up ^ `` Forbes ' Billionaires '' . Forbes . Jump up ^ `` Club Statement : Mel Morris Completes Purchase Of Derby County
'' . Derby County F.C. 4 September 2015 . Retrieved 4 September 2015 . Jump up ^ `` Sunday Times Rich List : Candy Crush tycoon Mel Morris and Duke of
Devonshire among Derbyshire 's millionaires '' . Derby Telegraph . 26 April 2015 . Archived from the original on 3 May 2015 . Retrieved 26 April 2015 . Jump up
^ `` Welcome To Shahid Khan '' . Fulham FC. 12 July 2013 . Retrieved 12 July 2013 . Jump up ^ `` Shahid Khan '' . Forbes .
</item 10>
=== END ITEM id=ff1cdf4d24a8bf27 ===

=== ITEM id=146fe189e515b776 ===
Gold sentences and facts:
Sentence 0: The Osborne 1 is considered the first true mobile computer by most historians . Adam Osborne founded Osborne Computer and formed the Osborne 1 in 1981 . The Osborne 1 had a five - inch screen , incorporating a modem port , two 5 1 / 4 floppy drives , a big collection of bundled software applications , and a battery pack . The computer company was a failure and did not last for very long . Although it was large and heavy compared to today 's laptops , with a tiny 5 '' CRT monitor , it had a near - revolutionary impact on business , as professionals were able to take their computer and data with them for the first time . This and other `` luggables '' were inspired by what was probably the first portable computer , the Xerox NoteTaker . The Osborne was about the size of a portable sewing machine , and could be carried on commercial aircraft . The Osborne 1 weighed close to 11 kg and was priced at $1795 .
  Fact 0.0: The Osborne 1 is considered the first true mobile computer by most historians.
  Fact 0.1: Adam Osborne founded Osborne Computer.
  Fact 0.2: Adam Osborne formed the Osborne 1 in 1981.
  Fact 0.3: The Osborne 1 had a five-inch screen.
  Fact 0.4: The Osborne 1 incorporated a modem port.
  Fact 0.5: The Osborne 1 had two 5 1/4 floppy drives.
  Fact 0.6: The Osborne 1 came with a big collection of bundled software applications.
  Fact 0.7: The Osborne 1 had a battery pack.
  Fact 0.8: The Osborne Computer company was a failure.
  Fact 0.9: The Osborne Computer company did not last for very long.
  Fact 0.10: Compared to today's laptops, the Osborne 1 was large and heavy.
  Fact 0.11: The Osborne 1 had a tiny 5-inch CRT monitor.
  Fact 0.12: The Osborne 1 had a near-revolutionary impact on business.
  Fact 0.13: Professionals were able to take their computer and data with them for the first time because of the Osborne 1.
  Fact 0.14: Other 'luggables' were inspired by the Osborne 1.
  Fact 0.15: The Osborne 1 was inspired by what was probably the first portable computer, the Xerox NoteTaker.
  Fact 0.16: The Osborne was about the size of a portable sewing machine.
  Fact 0.17: The Osborne could be carried on commercial aircraft.
  Fact 0.18: The Osborne 1 weighed close to 11 kg.
  Fact 0.19: The Osborne 1 was priced at $1795.
Sentence 1: The portable micro computer the `` Portal '' of the French company R2E Micral CCMC officially appeared in September 1980 at the Sicob show in Paris . The Portal was a portable microcomputer designed and marketed by the studies and developments department of the French firm R2E Micral in 1980 at the request of the company CCMC specializing in payroll and accounting . It was based on an Intel 8085 processor , 8 - bit , clocked at 2 MHz . It was equipped with a central 64K byte Ram , a keyboard with 58 alpha numeric keys and 11 numeric keys ( separate blocks ) , a 32 - character screen , a floppy disk : capacity = 140 000 characters , of a thermal printer : speed = 28 characters / second , an asynchronous channel , a synchronous channel , a 220V power supply . Designed for an operating temperature of 15 ° C to 35 ° C , it weighed 12 kg and its dimensions were 45cm x 45cm x 15cm . It provided total mobility . Its operating system was Prologue .
  Fact 1.0: The portable micro computer the 'Portal' of the French company R2E Micral CCMC officially appeared in September 1980 at the Sicob show in Paris.
  Fact 1.1: The Portal was a portable microcomputer designed and marketed by the studies and developments department of the French firm R2E Micral in 1980.
  Fact 1.2: The Portal was designed at the request of the company CCMC, which specializes in payroll and accounting.
  Fact 1.3: The Portal was based on an Intel 8085 processor.
  Fact 1.4: The Portal's Intel 8085 processor was 8-bit and clocked at 2 MHz.
  Fact 1.5: The Portal was equipped with a central 64K byte RAM.
  Fact 1.6: The Portal had a keyboard with 58 alphanumeric keys and 11 numeric keys in separate blocks.
  Fact 1.7: The Portal had a 32-character screen.
  Fact 1.8: The Portal had a floppy disk with a capacity of 140 000 characters.
  Fact 1.9: The Portal had a thermal printer with a speed of 28 characters per second.
  Fact 1.10: The Portal had an asynchronous channel and a synchronous channel.
  Fact 1.11: The Portal required a 220V power supply.
  Fact 1.12: The Portal was designed for an operating temperature of 15 °C to 35 °C.
  Fact 1.13: The Portal weighed 12 kg.
  Fact 1.14: The Portal's dimensions were 45cm x 45cm x 15cm.
  Fact 1.15: The Portal provided total mobility.
  Fact 1.16: The Portal's operating system was Prologue.
Sentence 2: The first laptop in the modern form was the Grid Compass 1101 , designed by Bill Moggridge in 1979 -- 80 , and released in 1983 . Enclosed in a magnesium case , it introduced the now familiar clamshell design , in which the flat display folded shut against the keyboard . The computer could be run from batteries , and was equipped with a 320 × 200 - pixel electroluminescent display and 384 kilobyte bubble memory . It was not IBM - compatible , and its high price ( US $ 8,000 -- 10,000 ) limited it to specialized applications . However , it was used heavily by the U.S. military , and by NASA on the Space Shuttle during the 1980s . The GRiD 's manufacturer subsequently earned significant returns on its patent rights as its innovations became commonplace . GRiD Systems Corp. was later bought by the Tandy ( now RadioShack ) Corporation .
  Fact 2.0: The first laptop in the modern form was the Grid Compass 1101.
  Fact 2.1: The Grid Compass 1101 was designed by Bill Moggridge in 1979-80.
  Fact 2.2: The Grid Compass 1101 was released in 1983.
  Fact 2.3: The Grid Compass 1101 was enclosed in a magnesium case.
  Fact 2.4: The Grid Compass 1101 introduced the now familiar clamshell design, in which the flat display folded shut against the keyboard.
  Fact 2.5: The Grid Compass 1101 could be run from batteries.
  Fact 2.6: The Grid Compass 1101 had a 320 × 200-pixel electroluminescent display.
  Fact 2.7: The Grid Compass 1101 had 384 kilobyte bubble memory.
  Fact 2.8: The Grid Compass 1101 was not IBM-compatible.
  Fact 2.9: The Grid Compass 1101 had a high price of US $8,000-10,000, which limited it to specialized applications.
  Fact 2.10: The Grid Compass 1101 was used heavily by the U.S. military.
  Fact 2.11: The Grid Compass 1101 was used by NASA on the Space Shuttle during the 1980s.
  Fact 2.12: The GRiD manufacturer subsequently earned significant returns on its patent rights as its innovations became commonplace.
  Fact 2.13: GRiD Systems Corp. was later bought by the Tandy (now RadioShack) Corporation.

Retrieved items:
<item 1>
History of laptops
IBM PC Convertible ( edit ) Also among the first commercial IBM - compatible laptops was the IBM PC Convertible , introduced in 1986 . It had a CGA - compatible
LCD and 2 floppy drives . It weighed 13 pounds ( 5.9 kg ) . Toshiba T1100 , T1000 , and T1200 ( edit ) Toshiba launched the Toshiba T1100 in 1985 , and has
subsequently described it as `` the world 's first mass - market laptop computer '' . It did not have a hard drive , and ran entirely from floppy disks . The
CPU was a 4.77 MHz Intel 80C88 , a variation of the popular Intel 8088 , and the display was a monochrome , text - only 640x200 LCD . It was followed in 1987 by
the T1000 and T1200 . Although limited floppy - based DOS machines , with the operating system stored in ROM , the Toshiba models were small and light enough to
be carried in a backpack , and could be run from lead - acid batteries .
</item 1>

<item 2>
History of laptops
Commodore SX - 64 ( edit ) The Commodore SX - 64 , also known as the Executive 64 , or VIC - 64 in Europe , was a portable , briefcase or suitcase - size ``
luggable '' version of the popular Commodore 64 home computer , and was the first full - color portable computer . The SX - 64 featured a built - in five - inch
composite monitor and a built - in 1541 floppy drive . It weighed 20 pounds . The machine was carried by its sturdy handle , which doubled as an adjustable
stand . It was announced in January 1983 and released a year later , at $995 USD . Kaypro 2000 ( edit ) Possibly the first commercial IBM - compatible laptop
was the Kaypro 2000 , introduced in 1985 . With its brushed aluminum clamshell case , it was remarkably similar in design to modern laptops . It featured a 25
line by 80 character LCD , a detachable keyboard , and a pop - up 90 mm ( 3.5 - inch ) floppy drive .
</item 2>

<item 3>
History of laptops
Both had LCDs , and could connect to optional external printers . The Dulmont Magnum , launched internationally in 1984 , was an Australian portable similar in
layout to the Gavilan , which used the Intel 80186 processor . Kyotronic 85 ( edit ) The year 1983 also saw the launch of what was probably the biggest -
selling early laptop , the Kyocera Kyotronic 85 . Owing much to the design of the previous Epson HX - 20 , and although at first a slow seller in Japan , it was
quickly licensed by Tandy Corporation , Olivetti , and NEC , who recognised its potential and marketed it respectively as the TRS - 80 Model 100 line ( or Tandy
100 ) , Olivetti M - 10 , and NEC PC - 8201 . The machines ran on standard AA batteries . The Tandy 's built - in programs , including a BASIC interpreter , a
text editor , and a terminal program , were supplied by Microsoft , and was written in part by Bill Gates himself .
</item 3>

<item 4>
History of laptops
Cambridge Z88 ( edit ) Another notable computer was the Cambridge Z88 , designed by Clive Sinclair , introduced in 1988 . About the size of an A4 sheet of paper
as well , it ran on standard batteries , and contained basic spreadsheet , word processing , and communications programs . It anticipated the future
miniaturization of the portable computer , and as a ROM - based machine with a small display , can -- like the TRS - 80 Model 100 -- also be seen as a
forerunner of the personal digital assistant . Compaq SLT / 286 ( edit ) By the end of the 1980s , laptop computers were becoming popular among business people
. The COMPAQ SLT / 286 debuted in October 1988 , being the first battery - powered laptop to support an internal hard disk drive and a VGA compatible LCD screen
. It weighed 14 lbs . NEC Ultralite ( edit ) Main article : NEC UltraLite The NEC UltraLite , released in mid-1989 , was perhaps the first `` notebook ''
computer , weighing just 2 kg ( 4.4 lbs ) .
</item 4>

<item 5>
History of laptops
Another significant machine announced in 1981 , although first sold widely in 1983 , was the Epson HX - 20 . A simple handheld computer , it featured a full -
transit 68 - key keyboard , rechargeable nickel - cadmium batteries , a small ( 120 × 32 - pixel ) dot - matrix LCD with 4 lines of text , 20 characters per
line text mode , a 24 column dot matrix printer , a Microsoft BASIC interpreter , and 16 KB of RAM ( expandable to 32 KB ) . Through at least the late 1980s
Epson continued to release laptops such as the L3s . GRiD Compass ( edit ) Main article : Grid Compass The first laptop in the modern form was the Grid Compass
1101 , designed by Bill Moggridge in 1979 -- 80 , and released in 1983 . Enclosed in a magnesium case , it introduced the now familiar clamshell design , in
which the flat display folded shut against the keyboard . The computer could be run from batteries , and was equipped with a 320 × 200 - pixel
electroluminescent display and 384 kilobyte bubble memory .
</item 5>

<item 6>
History of laptops
It was not IBM - compatible , and its high price ( US $ 8,000 -- 10,000 ) limited it to specialized applications . However , it was used heavily by the U.S.
military , and by NASA on the Space Shuttle during the 1980s . The GRiD 's manufacturer subsequently earned significant returns on its patent rights as its
innovations became commonplace . GRiD Systems Corp. was later bought by the Tandy ( now RadioShack ) Corporation . Dulmont Magnum / Kookaburra ( edit ) Main
article : Dulmont Magnum Another contender for the first true laptop was the Dulmont Magnum , designed Barry Wilkinson and Terry Crews Engineering Manager at
Dulmison in 1982 and released in Australia in 1983 It included an 80 × 8 display in a lid that closed against the keyboard . It was based on the MS - DOS
operating system and applications stored in ROM ( A :) and also supported removable modules in expansion slots ( B : and C :) that could be custom programmed
EPROM or standard word processing and spreadsheet applications .
</item 6>

<item 7>
History of laptops
Before laptop / notebook computers were technically feasible , similar ideas had been proposed , most notably Alan Kay 's Dynabook concept , developed at Xerox
PARC in the early 1970 s . One of the first reasonably portable computers was the Xerox NoteTaker , again developed at Xerox PARC , in 1976 . However , only 10
prototypes were built . Contents ( hide ) 1 Portal R2E CCMC 2 Osborne 1 3 Bondwell 2 4 Compaq Portable 5 Epson HX - 20 6 Grid Compass 7 Dulmont Magnum /
Kookaburra 8 Ampere 9 Tandy Model 100 10 Sharp and Gavilan 11 Kyotronic 85 12 Commodore SX - 64 13 Kaypro 2000 14 IBM PC Convertible 15 Toshiba T1100 , T1000 ,
and T1200 16 US Air Force 17 Hewlett - Packard Vectra Portable CS 18 Cambridge Z88 19 Compaq SLT / 286 20 NEC UltraLite 21 Apple 21.1 Macintosh Portable 21.
</item 7>

<item 8>
The history of laptops traces 1970s–1980s efforts to create compact, portable personal computers with the components and capabilities of desktop systems. Alan
Kay’s 1972 “Dynabook” prototype was an early vision. Sources cited include the TRS-80 Model 100/102, Commodore SX-64, Toshiba computers, and Compaq’s 1988
laptop.
</item 8>

<item 9>
History of laptops
Later models featured an Intel 80286 processor and a 20 MB hard disk drive . On the strength of this deal , ZDS became the world 's largest laptop supplier in
1987 and 1988 . ZDS partnered with Tottori Sanyo in the design and manufacturing of these laptops . This relationship is notable because it was the first deal
between a major brand and an Asian original equipment manufacturer . Hewlett - packard Vectra portable CS ( edit ) In 1987 , HP released a portable version of
their Vectra CS computer . It had the classic laptop configuration ( keyboard and monitor closes up clam - shell style in order to carry ) , however , it was
very heavy and fairly large . It had a full - size keyboard ( with separate numeric keypad ) and a large amber LCD screen . While it was offered with dual 3.5 -
inch floppy disk drives , the most common configuration was a 20 MB hard drive and a single floppy drive . It was one of the first machines with a 1.44 MB
density 3.5 - inch disk drive .
</item 9>

<item 10>
History of laptops
They also introduced the now - standard `` resume '' feature to DOS - based machines : the computer could be paused between sessions without having to be
restarted each time . U.S. Air Force ( edit ) The first laptops successful on a large scale came in large part due to a Request For Proposal ( RFP ) by the U.S.
Air Force in 1987 . This contract would eventually lead to the purchase of over 200,000 laptops . Competition to supply this contract was fiercely contested and
the major PC companies of the time ; IBM Corporation , To shins , Compaq , NEC , and Zenith Data Systems ( ZDS ) , rushed to develop laptops in an attempt to
win this deal . ZDS , which had earlier won a landmark deal with the IRS for its Z - 171 , was awarded this contract for its SupersPort series . The SupersPort
series was originally launched with an Intel 8086 processor , dual floppy disk drives , a backlit , blue and white STN LCD screen , and a NiCd battery pack .
Later models featured an Intel 80286 processor and a 20 MB hard disk drive .
</item 10>
=== END ITEM id=146fe189e515b776 ===
