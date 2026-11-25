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

This file has 3 items.

=== ITEM id=565e9e83158c2085 ===
Gold sentences and facts:
Sentence 0: Ketamine is primarily used as an anesthetic during surgical procedures — but in recent years it's been used as a remedy for treatment-resistant depression .
  Fact 0.0: Ketamine is primarily used as an anesthetic during surgical procedures.
  Fact 0.1: In recent years, ketamine has been used as a remedy for treatment-resistant depression.
Sentence 1: If it worked with the fish, they might then be able to figure out how the diet effects behavior and improve the effectiveness of the treatment for humans.
  Fact 1.0: If the treatment worked with fish, researchers might then be able to figure out how the diet affects behavior.
  Fact 1.1: If the treatment worked with fish, researchers might then be able to improve the effectiveness of the treatment for humans.

Retrieved items:
<item 1>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
Antibodies deal with viruses but fail to do so in HIV infection. By engineering bNAbs in a manner similar to mABs for cancer or vaccines (where antibody
reaction to neutered virus generates immune protection), HIV cure may be on the horizon.

Gilead’s several Phase-2 programmes with ART combination are in progress. Studies as recently as February 2023 presented proof of concept — 90 per cent
participants showed virological suppression at 26 weeks. Incidentally, Gilead’s Sovaldi and Harvoni were the first cures for Hepatitis C. The World Health
Organization (WHO) aims to eradicate the virus by 2030.

Alzheimer’s

The degenerative disease of the brain has proved to be a difficult challenge for pharmaceutical companies. Leqembi (lecanemab), developed by Eisai in
collaboration with Biogen, was the first and only approved drug for Alzheimer’s in July 2023.
</item 1>

<item 2>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
"A food craving or binge-eating is almost always tied to an emotional response," she said. "Chronic dieting, food shame, food accessibility and even early
childhood feeding experiences can all be tied back to food choices and disordered eating habits."

Although she believes that no single food is addictive, Palinski-Wade acknowledged that foods that are high in added sugar or added fats tend to be associated
with the strongest cravings and desire.

"These foods often trigger the release of feel-good chemicals such as dopamine in the brain, leading to cravings and potentially addictive-like eating
behaviors," she said.

"As you consume more and more of these foods, you build up a dopamine tolerance, meaning that you will have to eat even more of this food to experience the same
pleasure response," Palinksi-Wade said — which can contribute to the development of cravings and loss of control over eating behavior.

More research is still needed to fully understand the relationship between food and addiction, the experts all agreed.
</item 2>

<item 3>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
"The concentration of ketamine found in Matthew Perry’s blood was sufficient to cause loss of consciousness and lack of responsiveness to external stimulation,"
Lewis Nelson, chair of the Department of Emergency Medicine and chief of medical toxicology at Rutgers New Jersey Medical School, said in a statement provided
to Fox News Digital.

"This explains why he slipped under the water and did not awaken. Ketamine does not generally cause death from cardiac or respiratory effects, but rather from
associated injury."

MATTHEW PERRY CAUSE OF DEATH LISTED AS 'ACUTE EFFECTS OF KETAMINE'

Dr. Bankole Johnson, CEO and founder of Casa Privée in Miami , a concierge medical facility, offers ketamine infusion therapy to help patients manage various
conditions.

He told Fox News Digital that in his opinion, Perry’s death likely was linked to recreational ketamine use, although he did not treat or examine the actor.

Experts shared with Fox News Digital the following common myths and misconceptions about the drug.

Myth No.
</item 3>

<item 4>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
Previously, teenage patients who attempted suicide received no further contact from Children's Nebraska after they were discharged from the emergency
department, according to the hospital's website.

There was "no opportunity for Children’s Hospital staff to provide ongoing encouragement and support for patients struggling with mental health needs," as
stated on the site.

Nationally, about 27% of suicide survivors will make another attempt, said Schalley, but Children’s Nebraska’s numbers are now much lower.

"Thankfully, for the first three years of the program … 7% of enrolled patients are seen in a health care environment for a subsequent attempt," she told Fox
News Digital.

TIKTOK TELEHEALTH? TEENS PUT AT RISK BY SELF-DIAGNOSING MENTAL HEALTH ON SOCIAL MEDIA

On a national level, anywhere from 31% to 55% of pediatric patients who are taken to the emergency department for a suicide attempt will receive some sort of
follow-up within 30 days.
</item 4>

<item 5>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
Erin Palinski-Wade, a New Jersey-based dietitian with a focus on diabetes and nutrition, is skeptical that foods can be as addictive as drugs. (She was not
involved in the study.)

"Although foods rich in added sugar may stimulate the feel-good chemicals in the brain and become habit-forming, sugar itself is not addictive in the way
cocaine or another drug may be," she told Fox News Digital.

"Consuming sugar and then reducing or eliminating it from the diet will not result in withdrawal symptoms or side effects as would happen from a true
addiction," she continued.

"Food cravings are complex and tied into not just the nutrition profile of a food, but also the emotions and learned behaviors around eating."

APPLE A DAY — AND 4 OTHER GREAT FALL FOOD CHOICES THAT ARE HEALTHY, DELICIOUS AND FUN

Emotions, stress, overall metabolic health, sleep habits and learned feeding behaviors all shape people’s food choices, behaviors and responses, Palinski-Wade
said.
</item 5>

<item 6>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
Part of the reason that ultra-processed foods have addictive properties is that they deliver fats and carbs to the gut much faster than minimally processed
foods, the researchers wrote.

These foods also contain flavor and texture additives that could make them more addictive.

DEPRESSION RISK SPIKES FOR THOSE WHO EAT THESE UNHEALTHY FOODS, STUDY FINDS: ‘NOT A COINCIDENCE’

"There is converging and consistent support for the validity and clinical relevance of ultra-processed food addiction," said lead researcher Ashley Gearhardt, a
professor of psychology at the University of Michigan, in a press release announcing the study.

"By acknowledging that certain types of processed foods have the properties of addictive substances, we may be able to help improve global health."

Tanya Freirich, a registered dietitian nutritionist in Charlotte, North Carolina , who was not involved in the study, called out several warning signs of
addictive behaviors surrounding food.
</item 6>

<item 7>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
"Given the wide use of these drugs, these adverse events, although rare, must be considered by patients thinking about using them for weight loss."

The company provided the below statement to Fox News Digital on Friday, Oct. 6.

"At Novo Nordisk, patient safety is a top priority. We work closely with the U.S. Food and Drug Administration to continuously monitor the safety profile of our
medicines. The FDA-approved product labeling for Novo Nordisk’s GLP-1RA medicines indicated for use in weight management (Saxenda and Wegovy) includes
information about their potential side effects, including pancreatitis, acute gallbladder disease, ileus and delayed gastric emptying."

"Similar information is included in the product labeling for our GLP-1RA medicines indicated for the treatment of type 2 diabetes (Ozempic, Rybelsus and
Victoza)," the statement went on.
</item 7>

<item 8>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
But experiments published in BMC Biology with the Mexican cavefish, Astyanax mexicanus, may help reveal the link between diet and the brain.

Cavefish, an unexpected research model

For Masato Yoshizawa, geneticist and neuroscientist at the School of Life Sciences at The University of Hawai‘i Mānoa, the cavefish is a fascinating model
system. While these fish have unique adaptions for the harsh, lightless living conditions of limestone caves in northern Mexico — such as the loss of their eyes
— they also share several basic features with vertebrates, making experimental results with these fish applicable to other species. These features include the
brain region involved in memory and emotion, as well as many genes and the signalling pathways used to control them.

Humans and other vertebrates have more complexity in these systems, and experiments with the cavefish offer a way to strip it away to investigate fundamental
questions of biology. While working with cavefish, Yoshizawa noticed similarities in their behavior and that of patients with autism.
</item 8>

<item 9>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
One major limitation of the study is that it uses animal models that only represent certain aspects of disease and "do not present comorbidities or other
clinical diseases associated with aging, which are very common in Alzheimer’s patients," noted Desplats.

"Furthermore, the study was done under strictly controlled conditions that did not resemble the variety of people’s habits and environments," she added.

THE 8 BIGGEST ALZHEIMER’S DISEASE MYTHS — AND THE TRUTHS BEHIND THEM

The researchers hope that the public recognizes the importance of the circadian clock in regulating many aspects of health .

"Sustaining healthy sleep and eating habits that are more synchronized with the natural environment are vital in a society that is constantly exposed to food,
light and shift work," said Desplats.

Previous studies have supported this association between fasting and reduced Alzheimer’s risk.
</item 9>

<item 10>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications (Getty Images)

As with many other mental health disorders, therapy can often provide long-term relief from insomnia. The most effective of these therapies is Cognitive
Behavioural Therapy for Insomnia (CBTI),” says Dr Browning, describing it as a “complete cure” that a lot of people don’t know about. “Different from CBT for
depression or CBT for anxiety, CBTI works by addressing the negative thoughts and worries that perpetuate insomnia.”

Other avenues of treatment are also proving increasingly fruitful. A new drug called Quviviq was just approved by the NHS last week. Dr Browning is optimistic
about the “exciting” new medication, explaining that it acts in a very different way to other sleeping pills, which “knock you out” but cause people to develop
a tolerance.
</item 10>

<item 11>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
This story discusses suicide. If you or someone you know is having thoughts of suicide, please contact the Suicide & Crisis Lifeline at 988 or 1-800-273-TALK
(8255).

A hospital in Nebraska that matches teenage suicide survivors with a "caring contact" says the program has seen positive results in preventing young patients
from taking their own lives.

The first-of-its-kind program provides follow-up care in the form of handwritten, personalized notes from the hospital's social work staff after the teens are
discharged.

WHITE HOUSE ANNOUNCES NEW FUNDING FOR TEEN MENTAL HEALTH CRISIS: ‘WILL HELP SAVE LIVES’

The notes are sent one month, two months, three months, six months, nine months and one year after discharge — and the former patients are able to write notes
back as well.
</item 11>

<item 12>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Because fasting-mimicking diets can be done in longer periods, they can help the body go into higher levels of ketosis, Antoun noted, "which is wonderful for
the brain."

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

(Ketosis, which is when the body burns fat for energy instead of glucose, has been shown to have cognitive benefits.)

"Prolonged fasting can also help with leaky gut, which then can aid the brain," added Antoun.

In addition, intermittent fasting can help to control blood sugar , which can slow down the progression of Alzheimer's, the doctor said.

Over the past several decades, there has been a steady increase in diagnoses of Alzheimer's disease, Antoun noted.

"However, when we are able to detect neurodegenerative conditions early, we can take positive steps that may slow the process and mitigate any major side
effects of the disorder," he said.
</item 12>

<item 13>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
To nap or not to nap?

The siesta, a brief afternoon nap, is a traditional habit in Mediterranean countries. But scientists are undecided as to whether it is a healthy practice.

In the current study, specific napping habits were found to be associated with higher risk of mortality and cardiovascular disease. However, and similar to the
case of olive oil consumption, the researchers say they did not have detailed information on frequency and duration of naps or sleep structure, such as how
often and for how long, which they say weakens their conclusion.

“Moreover, reverse causation is also possible, with people napping because of subclinical conditions or other chronic diseases (for instance, if they had more
prevalence of sleep apnea), especially since napping faces a stigma in Anglo-Saxon countries,” said Sotos-Preito.

”Previous studies have found a J-shaped relationship — a smile-like curve — between napping and cardiovascular diseases,” said Sotos-Prieto.
</item 13>

<item 14>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
The researchers could not assess whether the condition was temporary or permanent.

Compared to another weight loss drug , bupropion-naltrexone, those who took a GLP-1 agonist were 3.67 times more likely to develop stomach paralysis, had a 9.09
times higher risk of pancreatitis and were 4.22 times more likely to have bowel obstruction.

OZEMPIC, WEGOVY AND ALL THOSE CRAZY, VIVID DREAMS: IS THERE A CONNECTION?

For the cases of stomach paralysis, the researchers could not assess whether the condition was temporary or permanent.

"There are reports from other journalists where they have met patients whose symptoms have not gone away despite stopping the drugs," study co-author Dr. Mahyar
Etminan, associate professor in the Departments of Ophthalmology & Visual Sciences and Medicine at UBC, told Fox News Digital.

There are medications that can be used to help with the condition, he added.
</item 14>

<item 15>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Like other medications, such as opiates or benzodiazepines, ketamine has the potential to be abused, Sullivan agreed.

"Ketamine should only be used within the confines of a close relationship between a patient and an experienced prescriber," he said, citing the position
reaffirmed by ASKP3.

Myth No. 5: It’s hard to kick the habit

"Weaning off ketamine is typically not too difficult, as the medicine has a half-life of three to four hours, and most is eliminated by the body in a single
day," said Johnson.

"Ketamine should only be used within the confines of a close relationship between a patient and an experienced prescriber."

Gradual dose titrations of ketamine — in which the amounts are slowly increased or decreased — are recommended for starting and stopping the medication, the
expert noted.

When ketamine becomes dangerous

Ketamine can become dangerous in a few ways, according to experts.
</item 15>

<item 16>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Surprisingly effective

Again, Yoshizawa was a bit skeptical. “I doubted it at first,” he said. “I didn’t think it was going to work, but it did.”

In the experiment, cavefish where fed the same ketogenic milk provided to human patients, albeit with a few modifications for fish consumption, and their
behavior was monitored. As a comparison, a type of A. mexicanus fish that lives in rivers and not caves were also tested.

The surface fish do not display the same autism like behaviors as their cave dwelling relatives. In the presence of other surface fish, individuals will begin
to follow each other and swim together, something rarely seen in cavefish, Yoshizawa said. The surface fish also do not do the repetitive behavior of swimming
in circles.

Using these fish as a comparison, Yoshizawa and his students watched and waited. Amazingly, after a month of the ketogenic diet, the cavefish began to act like
the more social surface fish.
</item 16>

<item 17>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
Popular weight loss drugs like Wegovy and Ozempic could increase the risk of stomach paralysis as well as several other serious gastrointestinal conditions ,
according to a study published Thursday in JAMA.

This was the first large epidemiological (disease-related) study to examine these adverse effects in non-diabetic patients using the drugs specifically for
weight loss, per a press release from The University of British Columbia (UBC).

The risk was linked to all semaglutides, a class of medications known as GLP-1 receptor agonists — including Ozempic (prescribed for diabetes management ),
Wegovy (prescribed for weight loss), Rybelsus (type 2 diabetes) and Saxenda (weight loss).
</item 17>

<item 18>
Could CRISPR cure HIV someday? (Live Science: The Most Interesting Articles, 2023-11-30T21:34:30+00:00)
cART therapies can suppress viral replication, but if the treatment is interrupted, "the dormant proviruses can reactivate, making a cure elusive,"
Herrera-Carrillo told Live Science.

CRISPR works by targeting and cleaving specific sequences of DNA from the genome; a "guide" leads CRISPR's famous "molecular scissors" to the targeted gene.
This either deactivates the gene or allows it to be removed and swapped for different DNA. Research groups believe this strategy could be effective in removing
latent HIV infections, because it can target the viral DNA embedded in the genome, rather than only stopping replication.

Using CRISPR for HIV gene therapy has shown promise in several test-tube studies, a 2020 review in The Journal of Clinical Investigation notes. Various groups
have been working to bring the therapies from test tubes to human patients — and that brings us to EBT-101.

Related: The world's 1st CRISPR therapy has just been approved.
</item 18>

<item 19>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
AS AI SHOWS UP IN DOCTORS' OFFICES, MOST PATIENTS ARE GIVING PERMISSION AS EXPERTS ADVISE CAUTION

"This model uses data from chest CT exams — often conducted for unrelated health issues, such as COVID or lung nodules — and identifies future heart disease
risk, which unfortunately even expert physicians can't discern from the scan," he said.

"The model helps cardiologists prevent potential heart attacks rather than treating them reactively."

Mayo Clinic has also developed AI models that predict the prognosis of patients with diseases such as colon cancer , detect risks of future cancers using
existing medical records, and predict 30-day hospital readmission or hospital-acquired infections, Patel said.

"AI can pinpoint details that make a significant difference in diagnosis and treatment."

"Our focus is not only to develop these AI models, but to ensure that the benefits of these models reach patients swiftly," he added.
</item 19>

<item 20>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
DOI: 10.1016/j.mayocp.2023.05.031

Feature image credit: Brooke Lark on Unsplash
</item 20>

<item 21>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“People who have insomnia are desperate. I treat patients who are suicidal, who have quit their jobs, whose lives are completely on hold because of their
sleeping issues. And they’re vulnerable. My clients have spent hundreds and hundreds of pounds on things that made no difference. If people are being mis-sold
expensive things and they’re being preyed upon, that’s a huge problem.”

Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications, teas, supplements, weighted blankets, and eye masks. “While supplements, exercise, and sleep hygiene provided some relief, the
true breakthrough came from addressing the root causes of my insomnia through therapy,” she tells me.
</item 21>

<item 22>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 22>

<item 23>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
More research is still needed to fully understand the relationship between food and addiction, the experts all agreed.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

The University of Michigan study was a review and compilation of many other studies' data, Freirich pointed out.

"Ideally, to see if the type of food made a difference, you could run a controlled study with people consuming one meal plan with ultra-processed foods and
another with minimally processed foods and then measure their reactions — thoughts about food, rated enjoyment of food, cravings and dopamine levels," she said.

Those who are struggling with disordered eating , binge-eating or uncontrolled food cravings should work directly with a therapist or registered dietitian
experienced in eating disorders, Palinski-Wade recommended.

"They can help you to better understand your behaviors and emotions around food and improve your relationship with food so that cravings and addictive behaviors
can decrease," she said.
</item 23>

<item 24>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
"Also, we could not look at risk with individual GLP-1 drugs, but this is probably a class effect of these drugs."

OZEMPIC-WEGOVY PILL MAY BE ON THE WAY: TRIAL SHOWS PROMISING RESULTS FOR NEW WEIGHT LOSS TABLET

The researchers also did not determine whether certain groups were at higher risk of this adverse side effect, but Etminan said he suspects that those with
preexisting GI conditions might be more susceptible.

California-based cardiologist Dr. Ernst von Schwarz, author of "The Secrets of Immortality," was not involved in the study but noted that it confirmed a higher
prevalence of pancreatitis, gastroparesis and bowel obstruction in patients on GLP-1 agonists.

"On the other hand, these drugs have been shown to reduce the risk of cardiovascular events as shown for Ozempic in 2016 in diabetics and in 2023 for Wegovy,"
he said in an interview with Fox News Digital.
</item 24>

<item 25>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
“Anyone with older parents should be urging them out for a walk five days a week – and when they have a companion, the walk also doubles as a social occasion,”
he explained. “If they can walk for longer than 20 minutes, even better. Our study showed that the risk of depression fell in proportion to the time spent
walking.”

As well as a daily walk, Laird recommends vitamin D and B12 supplements – both of which appear to moderate the chance of depression in older people. “Depression
matters,” he added. “Not just because it reduces quality of life, but because in older people there seems to be a link between depression and dementia which
we’re still unpacking.”

Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies. “I’d advise anyone with a parent in the
early stages of dementia to investigate all the available research trials. Start by joining Dementia Research, a nationwide register that matches volunteers to
research programmes.”
</item 25>

<item 26>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
As well, AI can help enable the delivery of personalized medicine and proactive preventative care, he noted.

"By analyzing individual genetic makeup combined with lifestyle and environmental factors, treatments can be tailored to the unique needs of each patient,"
Patel said.

"And by predicting potential health issues before they manifest, we can guide patients on preventive measures, fundamentally changing our approach from cure to
prevention."

Potential risks and limitations

AI in health care also presents challenges and limitations , Patel acknowledged.

Because AI is a branch of science, it requires rigorous evaluation before it can be applied by doctors, he said.

AI HEART SCAN AIMS TO CATCH BLOCKAGES YEARS BEFORE SYMPTOMS: ‘UNBELIEVABLE BREAKTHROUGH’

"This ensures that we're not just implementing technology for the sake of innovation, but are truly enhancing patient care in a tangible, evidence-based
manner."

AI models also run the risk of bias, he warned.
</item 26>

<item 27>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
“One way to start could be to emphasize the use of fresh, local products and ingredients in Mediterranean-style meals,” said Sotos-Prieto. “[But] changing
dietary habits and lifestyle is a gradual process, and affordability is also an important question.”

Other questions remain as to how governments should invest in introducing new healthy habits, which ones should be given preference, and how much change is it
enough to start seeing benefits.

The task is not simple since culture can define anything from ingredients and recipes to different social habits and work-life schedules. But Sotos-Prieto is
optimistic.

“I think that picking realistic goals, even if small, and sustaining them over time can make a difference [in public health],” concluded Sotos-Prieto.

Reference: Maroto-Rodriguez J. et al., Association of a Mediterranean Lifestyle With All-Cause and Cause-Specific Mortality: A Prospective Study from the UK
Biobank, Mayo Clinic proceedings (2023).
</item 27>

<item 28>
Could CRISPR cure HIV someday? (Live Science: The Most Interesting Articles, 2023-11-30T21:34:30+00:00)
Now, they're testing their EBT-101 targeting strategy in humans in an early-stage clinical trial that primarily focuses on the safety of the treatment. Initial
results from three treated patients showed no toxic effects or serious adverse events. All of the patients' HIV is currently suppressed with cART.

"The initial safety results are promising because no adverse outcomes have been observed to date," said Hope, whose lab studies the mechanisms behind HIV
infections. "But more time is needed to be sure because of the way genetic off-target mutations can take years to manifest into complications," he added.

Off-target effects of CRISPR refer to when the CRISPR molecule alters DNA at sites other than those targeted. These unintended snips have long been a worry for
researchers designing CRISPR treatments, so they'll be something to watch for with EBT-101, especially since the treatment targets multiple sites in the genome
.
</item 28>

<item 29>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Johnson, however, noted that ketamine is a "moderately addictive substance."

He told Fox News Digital, "Even long-term users typically only develop a behavioral or psychological dependence. Physical dependence can occur, but usually the
withdrawal signs are mild to moderate."

Dr. Sandhya Prashad, president of the American Society Of Ketamine Physicians, Psychotherapists and Practitioners (ASKP3) in Florida, said there is "an
extremely low risk of addiction, craving or withdrawal" when ketamine is used in low doses in a monitored clinical setting.

"Abuse of ketamine occurs when it is self-administered in high or frequent doses without supervision or monitoring, which leads to addiction ," she told Fox
News Digital.

Myth No 3: Ketamine can be self-prescribed and purchased online

Ketamine should only be obtained and administered by a licensed and experienced medical doctor, Johnson said.
</item 29>

<item 30>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
ULTRAPROCESSED FOODS WITH ‘FEEL-GOOD CHEMICALS’ COULD BE AS ADDICTIVE AS CIGARETTES AND DRUGS, STUDY SUGGESTS

"These can produce strong cardiac effects of a very high blood pressure and pulse, and can lead to serious medical complications," he said.

Myth No. 4: Intranasal or micro-dosing is safe for recreational use

While using small doses of ketamine may seem safer than the larger doses given via IV, Johnson warned that when this drug is self-administered, there is a
danger of exceeding safe limits.

"The total cumulative dose could end up being higher than a standard dose of a medically administered intravenous dose," he told Fox News Digital.

Intranasal use can also lead to "more aggressive drug-seeking" behaviors, Johnson warned, as the "high" of taking the ketamine is paired with cues in the user’s
environment — which are not present in a medical office.
</item 30>

<item 31>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
However, with a “traditional” approval for Eisai-partnered lecanemab for Alzheimer’s (compared to accelerated approval for adacanumab), Biogen has the first
approved treatment, putting adacanumab on the backburner.

Cognitive function improvement is a critical marker for Alzheimer’s treatment and lecanemab delivers lower degeneration compared to placebo on this count. The
treatment reduced clinical decline by 27 per cent at 18 months on clinical dementia scoring and on daily living scale delivered a 37 per cent improvement
compared to placebo. As is evident, the development is satisfactory but a lot more progress is left to be made on Alzheimer’s treatment curve.

Takeaways for Investors

Analysing pharma stocks in the US rests on three factors: drugs about to expire, drugs in ramping up stage and the pipeline.
</item 31>

<item 32>
Keep Labs’ smart storage redefines medication management (TechCrunch, 2023-10-04T19:24:25+00:00)
Somehow, Keep Labs won an award. Eventually, someone at the CTIA noticed Keep Labs’ marketing and pulled the prize. This set off a firestorm, eventually leading
the CTIA to soften its stance on cannabis-adjacent products — after all, cannabis is legal in Nevada, where CES is held.

This experience has had a lasting affect on Keep Labs. Look at the website. Cannabis is not mentioned. The Keep is now marketed with the catch-all term
“medicine.”

“Cannabis is still there,” he said, noting that the company is still on the same mission of harm reduction. “Cannabis was a great way to start a conversation
because everyone wants to talk about cannabis, but no one wants to talk about getting high.”

By simply saying “medicine,” the company feels like it found a good middle ground with its messaging. As Wilkins told me, Google and Meta heavily restrict the
term “cannabis” in advertising, which limits the marketing reach of such devices as Keep.
</item 32>

<item 33>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Previous studies have supported this association between fasting and reduced Alzheimer’s risk.

Last year, a study led by the USC Leonard Davis School of Gerontology in Los Angeles also found that mice that followed a fasting diet showed lower levels of
"plaques and tangles" in the brain, reduced brain inflammation and better performance on cognitive tests.

"Cycles of a fasting-mimicking diet (FMD) had strong effects against Alzheimer’s symptoms in two different types of Alzheimer’s mouse models," the lead author
of that study, USC biogerontologist Dr. Valter Longo, told Fox News Digital.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain."

"In mice, the effects of FMD cycles were very strong and wide, meaning they affected both learning and memory and also Alzheimer’s pathology peptides/proteins
(amyloid and tau)."
</item 33>

<item 34>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
“What I really wanted to do was work on trying to be a better me,” he says.

He discovered Resonance, a retreat center in Costa Rica on a hilltop overlooking the Pacific coast and the cloud forest of Monteverde.

To prepare his mind to enter a state of awareness, he was to eat only chicken, fish and salad for one month, with no oils and sugar added. Then he had to cut
out chicken and fish for the week preceding the ceremony. He also was told to practice yoga and breathing exercises that would be useful if he experienced
anxiety during the three ceremonies spread over a week.

In a tiki hut in a wooded, secluded area, a shaman presented each participant with a cup of ayahuasca brew, which is kind of like tea, but with benefits. Taste
is not one of them. Poyer says ayahuasca tastes like earth.

“It’s brownish and really thick,” he says. “It almost looks like oil, but it’s darker and thicker.
</item 34>

<item 35>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
ARTIFICIAL INTELLIGENCE HELPS DOCTORS PREDICT PATIENTS’ RISK OF DYING, STUDY FINDS: ‘SENSE OF URGENCY’

It’s important to protect data privacy and security as well, Patel noted, due to the large amounts of information that must be fed to AI models.

While AI's capabilities are "immense and ever-growing," Patel emphasized that AI’s role is to be an ally.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Humans caring for humans is the bedrock of health care — AI strengthens that foundation, but doesn’t replace it," he said.

"AI can offer insights and assist in decision-making, but human touch, intuition and empathy cannot be replicated by algorithms."

CLICK HERE TO GET THE FOX NEWS APP

While AI promises a "new frontier" in health care — Patel calls for a balance of "enthusiasm and caution."
</item 35>

<item 36>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
"I feel that it helps families know we truly do care about them when we’re assessing them – not just because it’s our job, but because we really want them to be
safe."

"Caring Contacts helps me make meaning from every interaction I have with the kids and families — that’s my ‘why,’ and what keeps me going," added Doescher.

To Schalley's knowledge, Children's Nebraska is the first pediatric emergency department to implement a program like Caring Contacts; but she hopes it will not
be the last.

"We have created a replication kit to provide other hospitals and organizations with information on how they can implement the program," she said, noting that
Children's Nebraska has presented the Caring Contacts concept at conferences and received interest from other facilities.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

Schalley suspects that the "intentionality" and deliberately low-tech communication between social workers and teens is another reason the program has worked so
well.
</item 36>

<item 37>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Buprenorphine, a drug that is prescribed to treat opioid use disorder , was in Perry’s system at the time of his death.

Guidelines to be published soon

ASKP3 announced that it will soon publish guidelines for safe use of the drug.

"In the wake of Matthew Perry’s autopsy report, we are committed to creating and publishing guidelines for at-home ketamine use, which should only be practiced
with a very specific patient profile and only in the context of a close relationship with a physician and in-office follow-up," the organization said in a
released statement.

CLICK HERE TO GET THE FOX NEWS APP

It is important for people to recognize the distinction between different uses of the drug, according to Prashad.

"In reality, the ways in which ketamine is used for anesthesia versus depression versus abuse are all very different in terms of dosage and frequency."
</item 37>

<item 38>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
"First, if a patient has access to use it at home for pain or mood, this can increase the likelihood of developing a physical or psychological dependence,"
Sullivan told Fox News Digital.

Experts recommend that home use of ketamine is only prescribed for select patients — who should be closely monitored with regular face-to-face office visits.

"Another way it can become dangerous is if a patient uses too large of a dose in an unsupervised setting, where they may put themselves at risk for accidents,
such as falls or drownings," said Sullivan.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

Mixing ketamine with other medicines can also be dangerous, Johnson warned.

For example, when mixed with opiates, it can have a stronger sedative effect, and combining ketamine with stimulants can worsen cardiovascular effects, he said.

In particular, ketamine mixed with buprenorphine can enhance sedation and lead to unresponsiveness, said Johnson.
</item 38>

<item 39>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
"By recognizing AI’s limitations and actively working to address them, we can harness AI's potential while safeguarding the core values of our health care
system."
</item 39>

<item 40>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
"The achieved weight loss using GLP-1 agonists, as well as the improved glucose control, seem to have significant benefits on cardiovascular outcomes, but
patients need to be informed about the relatively small incidence of abdominal side effects as seen in this cohort study," said von Schwarz.

"The benefits on cardiovascular risks , however, appear to outweigh the risks of side effects."

The U.S. Food and Drug Administration (FDA) lists "ileus" as a potential side effect of Ozempic.

Ileus is the inability of the intestine (bowel) to contract normally and move waste out of the body, according to Mayo Clinic.
</item 40>

<item 41>
Could CRISPR cure HIV someday? (Live Science: The Most Interesting Articles, 2023-11-30T21:34:30+00:00)
Scientists have been experimenting with how to use CRISPR to treat and potentially cure HIV since the gene-editing tool was developed.

In an ongoing clinical trial , researchers are testing whether just one dose of a new gene therapy that might effectively cure human immunodeficiency virus (
HIV ) infections is safe in humans.

The therapy, named EBT-101, involves using CRISPR-Cas9 gene editing to treat HIV. This potential treatment strategy has been studied in animal models since the
development of CRISPR-Cas9 in 2012. However, this is the first time such a gene-editing treatment for HIV has been tried in humans. The latest data from the
trial suggest that EBT-101 is safe at the doses tested, but we don't yet know if it cures HIV.

According to the United Nations Programme on HIV/AIDS (UNAIDS) , approximately 39 million people globally were living with HIV in 2022, and there were about
630,000 AIDS-related deaths that year, making HIV a continued public health burden.
</item 41>

<item 42>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
However, the challenge lies in deciphering which elements of the Mediterranean lifestyle contribute most significantly to improved health. Sotos-Prieto and her
team designed and developed a Mediterranean lifestyle index called MEDLIFE using a questionnaire that calculated an individual’s adherence to a Mediterranean
way of life. This index considers three primary components: food consumption, dietary habits, and physical activity, including rest, social habits, and
conviviality.

This type of analysis allowed the researchers to dissect the impact of each of these components, even drilling down to specific items like “hours of sleep” or
“socializing with friends”. Moreover, the MEDLIFE index could help capture synergistic effects of multiple components that may not be apparent when examining
individual components in isolation.

In their study, the team evaluated the impact of a Mediterranean-like lifestyle over the course of ten years in more than 110,000 participants located in the
UK.
</item 42>

<item 43>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
He said turning to Jesus for a deeper meaning and learning to not take his "first thoughts" for fact was crucial to his overall mental health.

"The Word of God is living and active, bringing strength, courage, wisdom, and protection," he said.

"God's thoughts and ways are higher than ours, but when we spend time, each day reading His Word and listening to the voice of the Good Shepherd, we are filled
with new thoughts that change the trajectory of our lives."

Bradley also shared his dos and don'ts of positive thinking.

He said it's not "denial and pretending that there are no challenges," he said.

"Positive thinking is not flattery and frivolous compliments."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Rather, positive thinking is realistic, full of truth and love — and aligned with heaven and scripture.

"Positive thinking is not self-consumption, self-absorption or selfishness. Positive thinking is not creating unrealistic expectations for other people and
putting extra stress on them to be perfect."
</item 43>

<item 44>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
Additionally, Freirich recommended filling the majority of meals with minimally processed foods , as consuming more of these may result in eating fewer
ultra-processed foods.

CLICK HERE TO GET THE FOX NEWS APP

"Lastly, speak to a dietitian about how to get started on changing your diet for your personal health goals," she said.

Fox News Digital reached out to the University of Michigan researcher for additional comments.
</item 44>

<item 45>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
Sabrina Schalley, director of care coordination at Children's Nebraska, a pediatric hospital and medical center in Omaha, Nebraska, told Fox News Digital that
Caring Contacts was inspired by a similar program for adult survivors of suicide.

In the fall of 2019, Schalley brought the idea of personalized follow-up care to the emergency department's social workers, "and they all liked the idea of the
Caring Contacts."

The program was set to begin on May 1, 2020 — and that timing would turn out to be rather providential.

"We had no idea when we started that COVID was around the corner and the impact that would have on children's mental health," said Schalley.

"We quickly became so grateful for the intervention."

TEEN GIRLS ARE STRUGGLING WITH MENTAL ILLNESS AT RECORD LEVELS, WITH MANY 'PERSISTENTLY SAD,' DATA REVEALS

In the years since the COVID pandemic began, children and teens are experiencing depression and anxiety more than ever before, according to Schalley.
</item 45>

<item 46>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Whenever you’re able, suggest audio-only meetings that don’t require screens so people can do things they would naturally do in an in-person setting, like move
around or look out a window," she advised.

"Walking away from our screens can reduce the urge to multitask, reduce eye fatigue and increase our capacity to move around."

Turning off the self-view option can also be helpful, Corcoran said — "it can be particularly distracting and draining because we’re not used to seeing our own
images so much."

As the use of video platforms is still relatively new for many professionals, Corcoran noted that the long-term effects aren’t yet known.

CLICK HERE TO GET THE FOX NEWS APP

"That being said, we know that constant feelings of fatigue, exhaustion and physical strain or tension do correlate with negative mental health outcomes," she
said.

"It’s important to take steps to reduce Zoom fatigue before it leads to things like chronic stress or burnout."
</item 46>

<item 47>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The goal of the restricted eating is to restore the circadian rhythm, the researchers said, which could help counteract the numerous disruptions that
Alzheimer’s patients face when it comes to sleep and schedule-related confusion.

HIDDEN BELLY FAT COULD SIGNAL ALZHEIMER’S DISEASE RISK 15 YEARS BEFORE SYMPTOMS SHOW UP, STUDY FINDS

"Our study emphasizes the power of feeding timing in aligning the circadian clock and its impact on the brain," said senior study author Paula Desplats, PhD,
professor in the Department of Neurosciences at UC San Diego School of Medicine, in a statement to Fox News Digital.

"We were hopeful to see some improvements in pathology, but did not expect such profound effects in reducing plaques and inflammation and improving memory," she
went on.

"The effects of time-restricted feeding on lower amyloid pathology were also measurable in blood using markers found in the clinic, which was another important
finding."

The researchers hope that these findings will lead to human trials.
</item 47>

<item 48>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
People in Britain could benefit from a key medical breakthrough next year. They may be given access to the first drugs ever developed to slow the impact of
Alzheimer’s disease.

The first of these medicines – lecanemab – was recently approved in the US and Japan, where treatments using it have already been launched. A second drug,
donanemab, is expected to follow soon, and next year it is anticipated that the UK medical authorities will consider both of them for approval in Britain.

The prospect has raised hopes that, after years of effort, scientists may be closing in on ways to directly tackle the UK’s dementia crisis. About a million
people are living with the condition in this country, and this is expected to rise to about 1.7 million by 2040 – with potentially grim consequences. Last year
dementia took the lives of 66,000 people in England and Wales, and it is now the leading cause of death in Britain, with Alzheimer’s accounting for two-thirds
of cases.
</item 48>

<item 49>
Dangerous 'superbugs' are a growing threat, and antibiotics can't stop their rise. What can? (Live Science: The Most Interesting Articles,
2023-10-01T11:00:46+00:00)
This cost might be a decrease in virulence or an increased vulnerability to antibiotics.

(Image credit: Graphic made by Olha Pohrebniak via Getty Images. Adapted by Live Science.)

To date, phage therapy has mostly been tested through a regulatory framework known as "compassionate use" in patients like the Brussels Airport bombing victim,
whose infections had no other treatment options. Phage therapy has shown success in these settings, and in a recent observational study of 100 patients who
received phages alongside antibiotics.

So far in clinical trials, though, phage therapy generally hasn't worked better than standard antibiotics or a placebo. Topline results from two recent trials
hint at the treatment's effectiveness in specific lung and foot infections, but the full results have yet to be released.

Success in future trials will be key to getting phages into the clinic, Turner said.
</item 49>

<item 50>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
They would follow each other in groups and ceased going round in circles. There were some other behaviors, such as attention to a specific task and sleeping,
that were unaffected, but overall the results were promising and according to Yoshizawa, suggest dopamine could be key to how the diet affects behavior.

“We suspect that due to this behavior variation, probably the target of the ketogenic diet is the dopaminergic system,” he said. The challenge now is teasing
out all the complex factors that may be involved.

Diet, environment, and genes – a complex recipe

According to Yoshizawa, there are two plausible ideas as to how the ketones produced by a ketogenic diet are acting on behavior. The first involves the
mitochondria, which use either carbs or fat to produce energy in our cells, and the other involves epigenetics, which simple refers to any non-genetic influence
which turns genes on and off.

Ketones are known to create detectable increases in gene expression in cells.
</item 50>

<item 51>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Being social, says Laird, is as important as trying new things: “Research shows that those who are more social are also more resilient to chronic conditions and
often have a stronger immune system.”

Anyone with older parents should be urging them out for a walk five days a week Dr Eamon Laird, health and ageing researcher

Depression among the elderly is on the rise and often goes unnoticed. Gerontologists now believe that it is one of the most common psychiatric disorders among
the elderly – with half of all cases going undiagnosed. When elderly parents become depressed (many of them come from a generation not accustomed to talking
about their mental health), it can be difficult to know how to respond. But Laird’s latest report suggests that a daily walk could help. His study of more than
4,000 older adults found that as little as 20 minutes of daily walking could cut the risk of depression by up to 43%.
</item 51>

<item 52>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
Key benefits of AI in health care

Patel said he views AI as a "powerful instrument" that helps magnify physicians’ capabilities rather than replacing them.

"One of AI's primary strengths is its ability to recognize patterns that might escape the human eye," he said.

"Whether it's intricate anomalies in medical imaging or subtle patterns in patient histories, AI can pinpoint details that make a significant difference in
diagnosis and treatment."

The technology also helps providers by sifting through and analyzing vast volumes of information, far more than what would be humanly possible, Patel said.

It also automates mundane and routine tasks, allowing health care professionals to redirect their focus to the patient, he noted.

AI TECH AIMS TO HELP PATIENTS CATCH DISEASE EARLY, EVEN ‘REVERSE THEIR BIOLOGICAL AGE’

Early risk detection is another key benefit.

"AI tools can predict potential health risks by analyzing a combination of genetic, behavioral and environmental factors, facilitating early interventions and
potentially saving lives," Patel said.
</item 52>

<item 53>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
There are medications that can be used to help with the condition, he added.

Although these complications were rare, the researchers found them concerning, given that millions of people are using these medications worldwide.

The number of people in the U.S. using GLP-1 agonists for either diabetes or obesity reached 40 million in 2022, they noted.

"These drugs are becoming increasingly accessible, and it is concerning that, in some cases, people can simply go online and order these kinds of medications
when they may not have a full understanding of what could potentially happen," said first author Mohit Sodhi, a graduate of UBC’s experimental medicine program,
in the release.

The number of people in the U.S. using GLP-1 agonists for either diabetes or obesity reportedly reached 40 million in 2022.

"Given the wide use of these drugs, these adverse events, although rare, must be considered by patients thinking about using them for weight loss."
</item 53>

<item 54>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
“Meals are something that happens around family and friends,” she continued. “The conviviality and social aspect are also characteristic. Physical activity is
important in the context of community life, and adequate rest including the typical short mid nap or siesta.”

Sotos-Prieto and researchers from the Universidad Autónoma de Madrid in Spain and the Harvard Chan School of Public Health in the USA are exploring the
feasibility and impact of adopting the Mediterranean diet and lifestyle among non-Mediterranean populations using locally available products and within their
own cultural contexts.

“We are seeing the transferability of the lifestyle [outside the Mediterranean] and its positive effects on health,” she added. The most significant perk? A 29%
lower risk of mortality in both men and women.

Breaking down the effects

Scientists stress that the Mediterranean diet should be understood more as a way of life than a mere dietary pattern with scientific evidence acquired over the
last four decades establishing its benefits on life expectancy, quality of life, and the prevention of several chronic diseases.
</item 54>

<item 55>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Conversely, the switch from winter to summer time was found to have no effect.

People frequently report an increase in SAD and depressive symptoms in winter (Getty/iStock)

“This study shows that the transition from summer time to standard time was associated with an increase in the incidence rate of unipolar depressive episodes,”
the study authors concluded. “Distress associated with the sudden advancement of sunset, marking the coming of a long period of short days, may explain this
finding.”

A drop in heart attacks

Not everyone is convinced, however. Charlie Morley, author of Wake Up to Sleep, cites a 2014 study by the College of Cardiology, which reported a significant
drop in heart attacks in the 24 hours immediately following turning back the clocks.

“Fascinatingly, each year when the clocks go back for Daylight Saving and 1.6 billion people across 70 countries gain an extra hour in bed, there is a 21 per
cent decrease in heart attacks the next day,” he says.
</item 55>

<item 56>
Keep Labs’ smart storage redefines medication management (TechCrunch, 2023-10-04T19:24:25+00:00)
Didn’t take your meds? The Keep knows and can trigger a smartphone notification on your or a companion’s phone.

To the founders, Keep is more than just a secure box; it’s a solution to harm reduction and medication management.

I recently spoke with co-founder, president and chairman Philip Wilkins about the company’s journey from a preorder success story to finding an enterprise
offering by partnering with Canadian healthcare providers.

From cannabis to medicine

Wilkins is an earnest entrepreneur. Spend a few minutes with him, and it’s clear he’s passionate about solving the problem of safeguarding medicine.

This was the second time I spoke with him. The first was back In 2019 when his company won an innovation award at CES 2020. This award was later rescinded
because the device was marketed as a cannabis storage device. In 2019, going into 2020, the trade group hosting CES had a strict but convoluted stance on
cannabis products. Somehow, Keep Labs won an award.
</item 56>

<item 57>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Ketones are known to create detectable increases in gene expression in cells. Pulling apart exactly how things like, diet, environment, genes and
neurotransmitters are linked is incredibly difficult but could reveal which pathways are best to target for autism treatments or could identify a specific
ketone which works more efficiently than others.

While the cavefish are not an exact model for human autism this work shows how these experiments shine light on the biology and evolution of these behavioral
conditions and offer new directions for researchers to follow.

Reference: Motoko Iwashita, et al. Metabolic shift toward ketosis in asocial cavefish increases social-like affinity, BMC Biology (2023), DOI:
10.1186/s12915-023-01725-9

Feature image credit: Masato Yoshizawa

Correction: This article was edited on November 2, 2023. Ryan Lee was originally described as working at the University of Hawai‘i Mānoa.
</item 57>

<item 58>
Could CRISPR cure HIV someday? (Live Science: The Most Interesting Articles, 2023-11-30T21:34:30+00:00)
Additionally, experts told Live Science that even though the ongoing trial hints that the therapy has a positive safety profile, we still don't know whether one
dose can effectively target latent HIV cells and whether it can control HIV in humans.

"It is crucial to approach this with caution," Herrera-Carrillo said. "While optimism about being on the right track is justified, it's important to recognize
the substantial work that still needs to be done."

The clinical trial will next test additional doses of EBT-101 for safety and then determine whether the virus stays suppressed when patients are taken off cART.
cART therapies pump the brakes on HIV replication, so the only way to determine whether the latent cell reservoirs have been disabled is to temporarily lift
those brakes.

Interruptions to HIV treatment are necessary to determine whether a patient is in remission — as has occurred in the few people cured of HIV — but in general,
purposeful treatment interruptions have been debated due to their inherent risks.
</item 58>

<item 59>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Based on these findings, Longo recommends that people talk to their neurologist about considering fasting-mimicking diets and other nutrition-based approaches
in support of standard-of-care drugs and therapies.

"Especially for Alzheimer’s disease, for which pharmacological interventions have had very limited efficacy, it will be very important to continue to study
fasting-mimicking diets and other dietary interventions that can make drugs more effective," he added.

Desplats, lead author of the UC San Diego study, told Fox News Digital that Longo’s studies are on a similar track, demonstrating the connection between
metabolism, fasting and brain health.

"The main difference is that we tested an approach that does not require calorie reduction, which may be more sustainable and safer for older people whose
metabolisms are already impacted by aging changes," she said.

COLORADO WOMAN WITH ALZHEIMER’S IS STILL VIBRANT AFTER MORE THAN 4 YEARS: ‘I REFUSE TO SURRENDER’

Dr.
</item 59>

<item 60>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
A Harvard study shares that optimism reduces the risk of dying prematurely — from overcoming trauma, making the most of your opportunities to overall life
satisfaction, positivity makes a difference."

WASHINGTON PASTOR ACKNOWLEDGES THE PAIN OF PUTDOWNS AGAINST THE CHRISTIAN FAITH: ‘ASK GOD FOR COURAGE’

The pastor said he wrote his most recent book, "The Power of the Second Thought," to help people gain habits to strengthen their mental health.

Bradley graduated from Dartmouth College and was playing professional soccer when he became ill — something that took 10 years to fully recover from. He said
finding hope during that time was challenging but is how he changes his "hopeless first thoughts."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain," he said.

FAITH DILEMMA IN FOCUS: ‘IT’S SUNDAY AND I HAVEN'T GONE TO CHURCH — WHAT DO I DO?'
</item 60>

<item 61>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
While on his working vacation in Maine, Kalmus invited NBC News for a short visit to the community to see the crisis through the eyes of Americans taking it as
seriously as anyone.

The Possibility Alliance is an 11-acre homestead that operates without power, running water or a sewer system. The Hughes family, which operates the alliance,
pays only one monthly bill for landline phone service. (Evan Bush / NBC News)

Detoxifying

About a decade ago, Kalmus ditched a promising academic career in astrophysics to study the physics of climate change instead.

He made other changes too. He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about
90%, according to his math.

It wasn’t a political statement, but a detoxification.

“Burning fossil fuels started to feel more and more disgusting,” he said.

Ethan Hughes, who operates the Possibility Alliance, rejects globalization and thinks capitalism is a ruinous symptom of “greed and separation.”
</item 61>

<item 62>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
The researchers recommend that regulatory agencies and drugmakers consider updating the warning labels for their products, which currently don’t include the
risk of gastroparesis.

"This is critical information for patients to know so they can seek timely medical attention and avoid serious consequences," said Sodhi.

The decision about whether to take the drug in spite of these risks will depend on each patient’s individual situation, the researchers said.

"This decision should be individually assessed," said Etminan. "There are patients where the benefit of the drugs might outweigh the risks (very obese
individuals), whereas in other situations (healthy individuals who just want to lose a few pounds), the risks might outweigh the benefits."

"This is critical information for patients to know so they can seek timely medical attention and avoid serious consequences."

The study did have some limitations, the researchers noted.

"We did not have access to medical charts to ascertain all subjects' medical histories," said Etminan.
</item 62>

<item 63>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
OZEMPIC DIABETES AND WEIGHT LOSS MEDICATION UNDER INVESTIGATION AFTER A FEW REPORTS OF SUICIDAL THOUGHTS

Stomach paralysis, officially known as gastroparesis, prevents the nerves and muscles in the stomach from moving food into the small intestine, which keeps
digestion from occurring, as described on Cleveland Clinic’s website.

In addition to stomach paralysis, the drugs were linked to a greater risk of pancreatitis (inflammation of the pancreas) and bowel obstruction, which prevents
food from passing through the small or large intestine, the release stated.

UBC researchers examined the health insurance claim records for approximately 16 million U.S. patients who were prescribed Ozempic, Wegovy or either semaglutide
or liraglutide medications across a 14-year span (between 2006 and 2020).

The researchers could not assess whether the condition was temporary or permanent.
</item 63>

<item 64>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
"To use ketamine, it is important to establish a clear diagnosis of severe depression or anxiety and/or post-traumatic stress disorder," he told Fox News
Digital. "This requires an expert and cannot be self-diagnosed properly."

"Abuse of ketamine occurs when it is self-administered in high or frequent doses without supervision or monitoring, which leads to addiction."

Those who think they may benefit from ketamine treatment should consult a doctor, Johnson advised.

"Ketamine is generally safe when administered by trained health professionals in a medical setting," Johnson said.

"In those settings, it is best delivered by the IV route, where the dose can be calculated accurately."

When purchasing ketamine outside a doctor’s prescription, there is also the risk of receiving a tainted product, experts warned. While people might assume
they're purchasing straight ketamine, the drug is often mixed with stimulants like cocaine or phencyclidine (PCP), Johnson noted.
</item 64>

<item 65>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
"Here at Children’s, our patients ages 11 and older are screening at risk of depression and/or suicide at nearly 30%," she said.

Each patient who receives a mental health assessment in the emergency department at Children's Nebraska is offered the chance to participate in Caring Contacts,
said Schalley.

About 98% of families agree to join the program, she said.

next Image 1 of 6

prev next Image 2 of 6

prev next Image 3 of 6

prev next Image 4 of 6

prev next Image 5 of 6

prev Image 6 of 6

"These are individuals who arrive due to suicide ideation or suicide gesture or who screen positive for suicide risk through the universal screening tool that
all emergency department patients ages 11 and older receive," Schalley told Fox News Digital.

Parents or legal guardians then sign a release form, which allows social workers at the hospital to keep in contact with the patient over the next year.
</item 65>

<item 66>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Pastor Jesse Bradley of Auburn, Washington runs Grace Community Church.

He's ministered to many, many people struggling with a variety of faith dilemmas and issues in their lives.

In this month of October 2023, as Americans grapple with a variety of issues, he addressed the notion of positive thinking — and shared with Fox News Digital
what it would "look like to change your thinking and shift in a new direction."

WASHINGTON PASTOR, ONCE GRAVELY ILL, SHARES HOW ‘THE POWER OF THE SECOND THOUGHT’ TRANSFORMED HIS LIFE

Said Pastor Bradley, "Psychologists report that we have in between 6,000 and 60,000 thoughts a day, based on the person and situation. The National Science
Foundation shares that up to 80% of our first thoughts are not helpful, even destructive."

What happens is that "when life is difficult, we can become flooded with thoughts that are drenched in fear, worry, stress, selfishness, impurity, selfishness
and lies.
</item 66>

<item 67>
How the conspiracy-fueled Epoch Times went mainstream and made millions (Hacker News, 2023-10-16T03:41:24+00:00)
(Unlike most major newspapers, The Epoch Times isn't audited by the two major independent collectors of circulation data.) The nonprofit has amassed a fortune,
growing its revenue by a staggering 685% in two years, to $122 million in 2021, according to the group’s most recent tax records.

Its editorial vision — fueled by a right-wing slant and conspiracy theories — is on display in recent reports on how “Jan. 6 Capitol Hill Security Footage
Challenges Key Narratives” and “Meteorologists, Scientists Explain Why There Is ‘No Climate Emergency.’” Its video series include a documentary-style film
alleging widespread vaccine injury and death and an exposé of an alleged world government agenda to harm farmers, cull the population and force survivors to eat
bugs.

What The Epoch Times lacks in standards, it makes up for in style and form, mirroring the aesthetics of journalism — a feature that’s attracted subscribers and
big-name supporters.

Anti-vaccine activist and presidential candidate Robert F. Kennedy Jr.
</item 67>

<item 68>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The researchers hope that these findings will lead to human trials.

The goal is to obtain funding to start a pilot study of patients next year.

"Accessible interventions like this have great translational potential, as they are available and are usually very well-tolerated," Desplats said.

The goal is to obtain funding to start a pilot study of patients next year.

"This may have special importance for the elderly , as time-restricted eating does not require limitation in calories or change in the diet, but may offer
important benefits ranging from metabolic and sleep regulation to potential cognitive improvement."

TIME-RESTRICTED EATING NO MORE BENEFICIAL THAN CALORIC RESTRICTION IN OBESE PATIENTS, STUDY SAYS

Adopting a new dietary pattern should not be a replacement for medical care , however, as Desplats emphasized the importance of patients' discussions about
options with their clinicians.
</item 68>

<item 69>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
This comes after several failed attempts, and one controversial false start in the recent years.

Alzheimer’s disease is characterised by amyloid beta deposits on brain cells leading to shrinking and loss of cognitive abilities. Despite a clear marker of
amyloid plaques and a target area to work on, pharma companies failed to address plaque removal or a statistical improvement in cognitive functioning after
plaque removal, till the recent approval. Pfizer, Eli lilly, Merck and many others faced failure in the space. The amyloid theory itself was in question.

Biogen’s adacanumab had a peculiar start. The company itself stopped development in 2019, but it surprisingly pursued a high-dose study into 2021. Later, FDA
approved adacanumab with an accelerated approval for high-dose version, which faced sharp criticism. Three advisers resigned over the approval of the drug that
could have cost $56,000 per annum to administer.
</item 69>

<item 70>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Ryan Lee was originally described as working at the University of Hawai‘i Mānoa. This was incorrect, and has now been corrected to say he works at Shriners
Hospital for Children Honolulu.
</item 70>

<item 71>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Until now doctors have only been able to prescribe medicines that help patients manage their symptoms, so the arrival of the first drugs that treat the actual
cause of the condition has been welcomed – although experts have warned that their use should be treated with some caution.

After decades of research, these are the first drugs to improve patients’ lives directly. If nothing else, they suggest we’re on the right road to tackling
Alzheimer’s

“The new drugs slow down the development of Alzheimer’s by six months to a year and are useful only for those in the early stages of the condition, so they are
certainly not miracle medicines,” said David Thomas, head of policy at Alzheimer’s Research UK.

“However, after decades of research, they are the first to improve patients’ lives directly, and that is a justifiable cause for excitement. If nothing else,
they suggest we are probably on the right road to tackling Alzheimer’s.”

This point was backed by neurologist Cath Mummery, of the Dementia Research Centre at University College London.
</item 71>

<item 72>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
This type of chart illustrates situations in which low values of one variable (in this case, a short nap duration) lead to a decrease in the values of the
second variable (incidence of cardiovascular events), while the trend reverses at higher values.

“Short naps, which range from 0 to 30 minutes, were associated with lower incident cardiovascular disease,” said Sotos-Preito. “An inverse association between
short naps and mortality and cardiovascular diseases was found in our previous publications.”

When asked for a definitive answer as to whether napping was indeed beneficial or not, Sotos-Prieto indicated that more research is needed in this regard.

Adapting to the lifestyle

The challenge remains in how to put these results into practice in other countries with different, deep-rooted customs. Cultural adaptation of beneficial
practices is essential to make the Mediterranean diet and habits more appealing and practical for diverse populations in non-Mediterranean countries.
</item 72>

<item 73>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
It has been used for more than a thousand years by Amazonian tribes. Ingesting it can induce a dreamlike feeling, an altered state of consciousness, mystical
experiences and euphoria.

In the summer of 2022, Aaron Rodgers revealed he had taken ayahuasca and credited it with helping him have two of the best seasons of his career for the
Packers.

Poyer had never heard of ayahuasca at the time. “My first thought was eff that,” Poyer says. “You crazy? I’m not doing that.”

The funny word kept showing up in different conversations on his social media feeds and podcasts. Poyer researched it and became intrigued.

Advertisement

After the difficult 2022 season, Rachel suggested Poyer take a “guys trip” after the season to check out. But “guys trips” usually involve alcohol, and he
didn’t want that. Sightseeing isn’t his thing. “What I really wanted to do was work on trying to be a better me,” he says.
</item 73>

<item 74>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Myth No. 1: Ketamine is a new treatment

In reality, the drug has been used for depression, post-traumatic stress disorder and pain for more than 10 years, according to Dr. Patrick Sullivan, medical
director of Initia Nova Medical Solutions in New Jersey .

The FDA approved ketamine for anesthesia in both humans and animals since 1970, the doctor noted.

After successful clinical trials , the FDA approved the drug in 2019 as a medication for treatment-resistant depression.

OZEMPIC AND WEGOVY WEIGHT LOSS DRUGS COULD HELP REDUCE ALCOHOL USE DISORDER SYMPTOMS, STUDY SUGGESTS

"There are hundreds of studies showing it to be safe and effective in the outpatient setting," Sullivan told Fox News Digital.

Myth No. 2: Ketamine is powerfully addictive

Ketamine is "legally recognized as having the potential for abuse and for both psychological and physical dependence," according to the American Addiction
Centers website.
</item 74>

<item 75>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
"Novo Nordisk stands behind the safety and efficacy of all of our GLP-1RA medicines when used consistent with the product labeling and approved indications."

CLICK HERE TO GET THE FOX NEWS APP

Novo Nordisk continued, "With respect to the study, as the authors acknowledge, the study has limitations, including potential confounding by indication and by
other factors."

The company added, "It is also important to note that the study analyzed data collected during the period between 2006 and 2020. During this time, Wegovy was
not on the market; Saxenda was first approved in December 2014. In addition, Victoza was FDA-approved in January 2010 and Ozempic was FDA-approved in December
2017."

Daniella Genovese of Fox News Digital contributed to this report.
</item 75>

<item 76>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
"It recognizes the impact of a child or adolescent telling their story to a stranger in an emergency room during a time of crisis," she said. "Before, it was
unlikely that the patient would hear from that social worker again."

"In our high-tech world, written words from a caring adult stand out and make a difference."

Now, that same social worker will send personal notes to the patient for a year, "offering unconditional caring and support along the way of their mental health
journey."

"The cards are tangible and many of the patients report hanging them up in their rooms, keeping them in their cars or backpacks, and reading them aloud to their
friends or family," said Schalley.

The cards could also work to save lives, as "most kids who make a suicide attempt go from thought to action in 10 minutes or less," said Schalley.

CLICK HERE TO GET THE FOX NEWS APP

"Imagine the power of having a card to reread … telling you that you are worthy from someone you met once," she said.
</item 76>

<item 77>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and
effectively. One major challenge they have faced is the problem of getting drugs to pass through the blood-brain barrier, which controls the movement of ions
and molecules from our bodies to our brains. That makes it difficult to get drugs into the central nervous system to tackle problems such as amyloid plaques.

“However, researchers are developing active transport methods for getting drugs across the blood-brain barrier,” said Mummery. “For example, we are now
exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact.”

Such developments will take years to realise, scientists caution, and a great deal needs to be done in the short term to deal with dementia.

“It is clear we have taken a step in the right direction but there is much more to do before we can have successfully dealt with these processes that are
happening to our brains,” said Thomas.
</item 77>

<item 78>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Reject and replace!"

He gave this list of examples:

Replace "People just slack and gossip at my job" with "I will work with all my strength, even when no one is looking"

Replace "There's no hope for my future" with "I am thankful for today and will keep doing the next right thing"

Replace "My children are annoying" with "My children are a gift from God"

Replace "I'll never have good friendships" with "I will pursue and support the people that I respect the most"

Replace "No one really cares about me" with "God is with me and loves me every day"

Said Bradley, "Johns Hopkins reports that there are links between positive thinking and health. People with a positive outlook are less likely to have a heart
attack or cardiovascular event."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain."

Also, "Research from the University of North Carolina points to positive people being more resilient and finding more solutions to problems.
</item 78>

<item 79>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
During prolonged periods of fasting, cells undergo a process called autophagy, which involves removing and recycling damaged cellular components, Antoun said.

CLICK HERE TO GET THE FOX NEWS APP

"This process is essential for maintaining proper cellular function and can help protect the brain from neurodegenerative diseases such as Alzheimer’s," he
explained.

More than six million Americans of all ages have Alzheimer's, and every 67 seconds, someone in America develops the disease, according to the Alzheimer’s
Association.
</item 79>

<item 80>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Positive thinking is not creating unrealistic expectations for other people and putting extra stress on them to be perfect."

Bradley said, "Positive thinking is a vital aspect of loving your neighbor, nourishing your mind, discerning between light and darkness, good and evil and right
and wrong."

"Positive thinking leads to taking action and improving situations. Positive thinking brings out the best in the people around you. Positive thinking rejects
excuses, takes responsibility, and overcomes setbacks."

The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive
thinking.

CLICK HERE TO GET THE FOX NEWS APP

"Jesus exudes a trustworthy positivity that is full of meaning, faithfulness, and transformation," he noted.

"God renews our thinking and vision, empowers us to make intentional choices about our focus and perspective and trains us how to win the battle of our minds,"
he added.
</item 80>

<item 81>
Could CRISPR cure HIV someday? (Live Science: The Most Interesting Articles, 2023-11-30T21:34:30+00:00)
Related: The world's 1st CRISPR therapy has just been approved. Here's everything you need to know

HIV infects immune cells in the body. (Image credit: KATERYNA KON/SCIENCE PHOTO LIBRARY via Getty Images)

All about EBT-101

According to a recent presentation at the annual meeting of the European Society for Gene and Cell Therapy in Brussels, EBT-101 uses multiple guides to target
multiple sites in the genome and snip out large sections of latently integrated HIV DNA. This stops HIV from replicating.

Kamel Kahlili , a professor of neurovirology and gene editing at Temple University and co-founder of Excision BioTherapeutics , has been working with the
company to develop EBT-101 for a decade. In 2020 and 2023 , Khalili and his team published reports that showed that EBT-100, a precursor to EBT-101, safely
targeted and removed HIV DNA in infected primates.
</item 81>

<item 82>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
The fish avoid their peers and engage in repetitive behavior, such as swimming in circles.

“Many people first doubted that the fish have an autism-like state; I also doubted it at first,” said Yoshizawa. But as he soon found out, even patterns of gene
regulation resembled autistic patients.

Believing that cavefish might be useful for the study of autism, Yoshizawa approached his colleague Ryan Lee, a physician working with autism patients at
Shriners Hospital for Children Honolulu. “When I showed him the video of the cavefish he immediately said, ‘yes, that is very similar to what I observe in my
patients’,” recounted Yoshizawa.

Lee, who has experimented with ketogenic diets to improve autism symptoms with some success suggested they try it with the fish. In humans, not all patients
receiving the ketogenic diet improve, but some do. If it worked with the fish, they might then be able to figure out how the diet effects behavior and improve
the effectiveness of the treatment for humans.
</item 82>

<item 83>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
"One hundred percent of patients enrolled in Caring Contacts are receiving follow-up," noted Schalley.

"The program normalizes the struggles and challenges of mental health and stresses the importance of ongoing treatment."

Schalley credits the success of Caring Contacts to a multitude of factors, she told Fox News Digital.

"The program normalizes the struggles and challenges of mental health and stresses the importance of ongoing treatment," she said. "It provides positive coping
skills and words of encouragement."

Plus, the program "provides the ability for the youth to maintain a positive, trusting relationship with the social worker," said Schalley.

TEEN GIRLS DEALING WITH MENTAL HEALTH CRISIS NEED ‘CONNECTION’ MOST, DOCTORS SAY

Unexpected beneficiaries of the Caring Contacts program are the staff members who write the cards.

Participating in Caring Contacts has "provided [staff] with an avenue to receive responses from the youth regarding updates on how they have progressed and hear
feedback on their experience from the initial encounter," said Schalley.
</item 83>

<item 84>
How the conspiracy-fueled Epoch Times went mainstream and made millions (Hacker News, 2023-10-16T03:41:24+00:00)
Bauer, an assistant journalism professor at the University of Alabama who studies conservative media. “An entire new kind of ecosystem of Covid deniers and
skeptics overlapped with right-wing media and were citing and drawing heavily upon The Epoch Times.”

The Epoch Times’ subscription page began hosting glowing testimonials from Steve Bannon, Glenn Beck and the far-right Arizona congressman Paul Gosar. The
organization became a reliable source for misinformation around Covid, its treatments and the vaccines.

The Epoch Times was also an early and aggressive promoter of election misinformation, according to the Election Integrity Partnership, a coalition of
researchers that documented misinformation and the 2020 vote. The group cited the news organization as a “repeat spreader” of false and misleading voter fraud
stories as well as a major promoter of debunked conspiracy theories around Dominion voting machines and the “Stop the Steal” movement, aimed at overturning the
election results. Months after the election, The Epoch Times refused to acknowledge the results.

With the new bedfellows came a new revenue stream.
</item 84>

<item 85>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Kalmus, trailed by a reporter, buzzed around the homestead like a giddy kid, hiking back and forth to a home in an eco-village near the Possibility Alliance
where he was living and working on his research. He tramped down a forest trail in bare feet, paused work for a midday dip in a shady stream and never changed
his T-shirt.

He ground his coffee with a mortar and pestle, thumbed through a book of Enneagram personality types on the porch with his son Zane and listened to Ethan’s
stories about sailing across the Atlantic in a cargo ship to avoid the carbon emissions of flying.

“I feel met by Peter,” Sarah Wilcox-Hughes said of Kalmus. “I don’t feel like there are many people prioritizing the needs of the planet as he is and we are.”

Kalmus found life in Belfast, and at the Possibility Alliance, slower, richer and more connected to the natural world.
</item 85>

<item 86>
Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study (FOX News - Health, 2023-10-07T09:30:42+00:00)
CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

In August, a Louisiana woman sued Novo Nordisk, the manufacturer of Ozempic and Wegovy, and Lilly, which makes another GLP-1 agonist, Mounjaro, claiming that
the companies "downplayed the severity of gastrointestinal events caused by Ozempic and Mounjaro — never, for example, warning of the risk of gastroparesis
(‘paralyzed stomach’) or gastroenteritis."

The woman, who used Ozempic for more than a year before switching to Mounjaro last month, alleged she was "severely injured as a result" of using both drugs,
the suit stated.

In a statement at the time, a spokesperson for Novo Nordisk told FOX Business that gastrointestinal events "are well-known side effects of the GLP-1 class" and
"are mild to moderate in severity and of short duration."
</item 86>

<item 87>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
Treatment regimens are now not just restricted to glycemic control. Novo Nordisk’s Semaglutide, a leading GLP-1RA regimen, has shown in studies that it can
reduce an adverse cardiovascular event by 20 per cent in overweight patients. In Type-2 diabetes patients, Sema’s study to show improved outcomes in chronic
kidney disease was successful even on interim analysis.

Semaglutide, branded as Wegovy, showed a 17-18 per cent weight loss sustained over 68 weeks’ trial for obese patients treated with Wegovy. This has created a
flutter as Pfizer, Eli Lilly and others have joined the race in the category.

Convenience is another important dimension on which anti-diabetes treatments place a significant weight. Once-a week insulin, Ozempic, has been approved and
once-a-week long-acting basal analogue is in the works.
</item 87>

<item 88>
Keep Labs’ smart storage redefines medication management (TechCrunch, 2023-10-04T19:24:25+00:00)
Wilkins continues to talk to customers. The company has an active beta program with 150 users, and he calls the top 10 users weekly to better understand their
usage. He’s discovered people are now using their Keep for other critical items like keys, passports and cash. He believes these additional use cases show users
trust the device.

With Wandzura leading the company and with gobs of customer feedback, Keep Labs started exploring an enterprise offering focused on medication adherence along
with harm reduction.

The company became part of the McKesson Digital Health Network in Canada to provide real-time data about whether patients are taking their medication. Keep Labs
also has a partnership with Savvy Cooperative to give away free Keep devices to patients who are living with chronic conditions. These partnerships provide an
invaluable feedback loop of front-line patients interacting with their devices.

“Let’s make it super easy to deploy to individuals who need support,” Wilkins said.
</item 88>

<item 89>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
After Matthew Perry’s cause of death was listed as "acute effects of ketamine" on Friday, experts are speaking out about misconceptions surrounding the drug.

(Other conditions that contributed to Perry's death, according to the autopsy report, included "coronary artery disease [and] buprenorphrine effects." Also,
"prescription medications and loose pills" were found at the residence, the report said.)

Ketamine is primarily used as an anesthetic during surgical procedures — but in recent years it's been used as a remedy for treatment-resistant depression .

WHAT IS KETAMINE, THE DRUG THAT KILLED MATTHEW PERRY ON OCTOBER 28?

The Alcohol and Drug Foundation (ADF) describes ketamine as a "dissociative drug," which means it causes people to feel "separated or detached" from their
bodies or physical surroundings.

While it is also used illegally as a recreational drug , experts say ketamine is generally not dangerous when used as prescribed.
</item 89>

<item 90>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Both drugs are to be considered for approval in the UK next year. The Medicines & Healthcare products Regulatory Agency (MHRA) will first decide if they are
safe and effective, then the National Institute for Health and Care Excellence (Nice) will rule on whether they offer value for money.

Both medicines are expensive – lecanemab costs about $25,000 (£19,700) a year – and are given by regular intravenous infusions. “That is a challenge from a
health-service point of view, because you need to find the space and time to put someone in an infusion suite to treat them,” added Thomas.

However, the main problem facing doctors is the difficulty involved in pinpointing dementia not just in its early stages but even in its later phases. Most
cases are first presented to GPs, who will then refer patients to memory clinics for a dementia test. However there are long waiting times – up to two years on
average – for appointments at these centres.
</item 90>

<item 91>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
It’s been proven that what people eat can help prevent or slow Alzheimer’s disease — but what about when they eat?

Participating in intermittent (time-restricted) fasting could lead to a reduced risk of cognitive deterioration, a recent study published in the journal Cell
Metabolism suggests.

Researchers at University of California San Diego School of Medicine adjusted the feeding schedule of certain groups of mice so that they only ate within
six-hour windows each day.

ALZHEIMER’S BLOOD TEST COULD HIT THE MARKET IN EARLY 2024, RESEARCHERS SAY

(This would be equivalent to 14 hours of fasting for humans, the researchers noted.)

Compared to a control group of mice that ate on demand, the fasting mice showed improvements in memory, were less hyperactive in the evenings and had fewer
sleep disruptions .

They also were shown to have less buildup of amyloid proteins in the brain, which is a hallmark of Alzheimer’s disease.
</item 91>

<item 92>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
Could a craving for salty chips actually be a sign of addiction?

A new study from the University of Michigan suggests that could be the case.

Researchers reviewed 281 studies from 36 different countries, finding that 14% of adults and 12% of children showed signs of addiction to ultra - processed
foods , according to the Yale Food Addiction Scale (YFAS).

ULTRA-PROCESSED FOOD CONSUMPTION LINKED TO HIGHER RISK OF DEATH FROM OVARIAN, BREAST CANCERS: NEW STUDY

That’s close to the addiction levels seen for alcohol and tobacco, noted the study article, which was published in the journal BMJ.

Ultra-processed foods are those that are high in carbohydrates and fats, such as candy, ice cream and potato chips.

The study suggests that some people’s consumption of these foods may meet the criteria for substance use disorder , resulting in the "biopsychological
mechanisms of addiction and clinically significant problems."
</item 92>

<item 93>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 93>

<item 94>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
Mackenzie Parks, a licensed clinical social worker at Children's Nebraska, said the families she encountered were "touched and flattered" that the hospital
would offer follow-up support.

"The letters I’ve received in return are really telling about the impact the program has on their well-being and hope for the future," she told Fox News
Digital.

"I also feel that it helps families know that we truly do care about them when we’re assessing them — not just because it’s our job, but because we really want
them to be safe," Parks added.

TEEN GIRLS SPEND MORE TIME ON ‘SENSITIVE’ SOCIAL MEDIA CONTENT THAT CAN HARM MENTAL HEALTH, REPORT SAYS

Eva Doescher, a licensed clinical social worker in Nebraska, shared similar feedback on her past Caring Contacts experiences.

"Caring Contacts gives me a chance to slow down and learn from the powerful, short moments that I get to connect with patients and their families," she told Fox
News Digital.
</item 94>

<item 95>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
More than 20 countries have also added nutrition labels to these foods, she wrote.

"It’s essential to understand the addiction to these ultra-processed foods globally, particularly in low- and middle-income countries," Gearhardt said in the
release.

"Chronic dieting, food shame, food accessibility and even early childhood feeding experiences can all be tied back to food choices and disordered eating
habits."

"It will take courageous action to change these and other economic and structural factors that drive people toward ultra-processed foods."

Freirich noted that many countries have banned the production of foods with certain additives that are shown to be detrimental.

6 WAYS TO TAKE YOUR SALAD FROM BORING TO BOUNTIFUL: HEALTHY FATS AND SMART PREP ARE JUST SOME OF THE SECRETS

"The USDA and FDA could make similar steps to ensure that foods are safe for consumption," she recommended.
</item 95>

<item 96>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
When thinking about treatments for conditions like autism spectrum disorder, diet is likely not high on the list. But clues from an eyeless species of
cave-dwelling fish point to a link between a trendy weight loss diet, the brain, social behavior, and autism.

The ketogenic diet is designed to induce a physiological state called ketosis. This happens when the body no longer has enough carbohydrates to burn for energy
and switches to fat as an energy source. Breaking down fat for energy produces a group of molecules called ketones, which travel throughout the body.

High in protein and fat but low in carbs, the ketogenic diet is popular for its weight loss effects but also appears to have some neuroprotective and
anti-inflammatory effects too. Children with epilepsy are sometimes placed on this diet because it helps reduce the frequency of seizures.

The exact mechanism behind these neurological effects is still a mystery and there have been reports of long-term side effects in people trying the diet outside
of clinical settings.
</item 96>

<item 97>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Joseph Antoun, CEO and chairman of the nutritech company L-Nutra in California, was also involved in the USC study.

"In these studies, mice fed the fasting-mimicking diet showed improved cognitive performance, motor learning and both short- and long-term memory, and promoted
the growth of new neurons in the brain," he told Fox News Digital via email.

"Fasting-mimicking diets have the potential to drastically reduce many aging and disease risk factors by reducing neuroinflammation or superoxide production in
the brain."

Fasting can help the brain function in a myriad of ways, said Antoun.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain," he told Fox News Digital in an emailed
statement.

Time-restricted eating can also improve the quality of sleep, he noted, which can help the brain recover better.
</item 97>

<item 98>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
With positive impact on non-glycemic aspects as well including weight loss, cardiac events and kidney functions, GLPs will witness more product innovation and
R&D commitment in the short term.

GLP-1 is a normal peptide released by the intestine in response to a meal. This binds to GLP-1 receptors in the pancreas and activates insulin production, which
is impaired in Type-2 diabetes patients. GLP-1 receptor agonists are external agents that bind to the GLP-1 receptors and elevate insulin secretion by the
pancreas. But, GLP-1 receptors are also found in kidney, lung, heart, skin, immune cells, and several other tissues. This allows GLP-1RA to attain a multi-modal
result, which improves cardiac and renal outcomes, and aids weight loss slower gastric emptying by binding to intestinal receptors. This feature separates
GLP-1RA from other modes.

Treatment regimens are now not just restricted to glycemic control.
</item 98>

<item 99>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
These include thinking about food all the time or obsessively, craving foods when not hungry, craving foods as a coping mechanism, eating even after feeling
full, having reduced control over intake, experiencing symptoms of withdrawal, and continuing to eat certain foods despite negative consequences.

SUGAR SUBSTITUTES NOT ADVISED FOR WEIGHT LOSS OR DISEASE PREVENTION, SAYS WORLD HEALTH ORGANIZATION

"While you can enjoy food thoroughly, being addicted to food starts to veer into an unhealthy territory with detrimental effects for the rest of your life,"
Freirich told Fox News Digital.

"For example, overeating past the point of comfort or to the point of causing digestive distress, or struggling to concentrate on other topics besides food and
meals, can be signs of a food addiction."

Gearhardt noted that 103 countries have passed "sugar-sweetened beverage taxes," and several others have additional taxes for ultra-processed foods.

More than 20 countries have also added nutrition labels to these foods, she wrote.
</item 99>

<item 100>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
The scientists found that adoption of the Mediterranean lifestyle was indeed feasible and could be adapted to local contexts, such as availability of
ingredients, their affordability, local culinary traditions, and food preferences. Participants were found to have lowered their risk of cancer and
cardiovascular disease relative to the general population.

“We found that the three blocks of the MEDLIFE were independently associated with lower […] mortality,” explained Sotos-Prieto. “And within block 3 [physical
activity, rest, social habits, and conviviality] we found that participating in collective sports, limiting sedentary activities, and having adequate hours of
sleep (6-8 hours) were associated with lower mortality.”

However, there were some variables that could not be taken into account due to a lack of information. This was the case for olive oil consumption, a prominent
component of the Mediterranean diet previously associated with protective roles in many chronic diseases. “It is probable that we are underestimating the
Mediterranean lifestyle association,” said Sotos-Prieto.

To nap or not to nap?
</item 100>
=== END ITEM id=565e9e83158c2085 ===

=== ITEM id=89402724fcd73d2d ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their liability.
  Fact 1.0: Sportsbooks might adjust the odds to balance their liability if a significant amount of money is placed on one team or participant.

Retrieved items:
<item 1>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We don’t want to talk too much about post-game stuff, but that was a tough one, probably the toughest I have had as a coach. He took it the way I expected him
to. He’s a champion.”

Papali’i said the All Blacks were intent on sending their leaders out on a high.

“There’s a few leaders in the team, it’s going to be their last game in the jersey, so it’s a big game for us and even the other players that are staying,
knowing that we want to send those boys out on a high,” he said.

“How much they put into the jersey. We are standing on their…what’s that quote? It’s standing on the shoulders of giants.”

The generational change isn’t just occurring on the field but off it, too.

Advertisement

Foster will depart the All Blacks set up after 12 years, the last four as head coach.
</item 1>

<item 2>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
“Firstly, you’ve never heard us say we’re favourites,” he said.

‘We know these games are do or die. We’re going to have to improve our performance. The stage gets bigger at this time in the tournament and you’ve got to grow
your game. That’s our strong focus. The best team on the night wins – Argentina has done that to us.”

Cheika however was happy to remind the All Blacks who were favourites.

“”New Zealand has always been an example in rugby, a benchmark. They make you think of high-level skills in a really open game but there are always threats in
the lineout, mauls and rucks,” said Cheika, whose Wallabies lost in the 2015 final.

“You have to take the moments you are in the game and see them coming at you at different times.

Advertisement

“In the game in 2020 there were things you could pinpoint.
</item 2>

<item 3>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“It’s the life of this business, it’s a tough game. It’s tough when you’re trying to get your performance right. It’s a tough game when people around you see
things differently, but we’ve learned a lot about ourselves.

“We take massive pride in the jersey and making sure we reach the levels we want to. The rest of that is probably a conversation for another day.

“It hasn’t altered our vision of where we want this team to be. I know we’ve surprised a few people but I don’t think we’ve surprised ourselves.”
</item 3>

<item 4>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“To inspire is not about results, it’s not just one player who played well; no, it’s about our behaviour in general, our ability to be ready to come back from
difficult times and seizing our opportunities.

“There are not a lot of opportunities in life – us playing in a semi-final is even rarer and we want to have no regrets.”
</item 4>

<item 5>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
The All Blacks are 18-point favourites for the opening semi-final.

It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.

“The past sets us up beautifully for both teams,” Foster said.

“We do know each other but we don’t know each other terms of a Rugby World Cup, so at this stage in the tournament it’s new territory for us and as we’ve seen
World Cups are very different.”

“We have learnt to greatly respect Argentina. They have a rich history of overachieving at the World Cup and have done a fantastic job to get here at the same
level we are, so it is going to be a heck of a game.”

Advertisement

Foster said the All Blacks, who suffered a shock semi-final loss to Eddie Jones’ England in 2019, never consider themselves favourites.

“Firstly, you’ve never heard us say we’re favourites,” he said.
</item 5>

<item 6>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“There were a few tears and I think that’s awesome to see, players showing vulnerability, just opening up about how much this tour and this sort of game means.



“There were some (people) saying this is probably one of the most important games of All Blacks history. (The) All Blacks went through a bit of a hard path to
get here, media, friends, family even criticising us and all that, and we stuck together. We weren’t even said to be in the final and now look at us, we’re
here.

Advertisement

“Different players take it differently and some of the older players probably were hit a little bit harder and it was good to see just a bit of emotion and the
want from them. Seeing that gives all the other players an extra two, three per cent to think ‘OK, we are going to the well and we’ve got to give it our all’.”

The All Blacks will experience a changing of the guard next season.
</item 6>

<item 7>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Pressure? The South African laughs at the notion of what pressure is compared to the inequality and challenges the Rainbow Nation continues to experience.

Advertisement

The question is: how can New Zealand’s All Blacks withstand and overcome the hitherto unshakable wave of momentum and spirit the Springboks play with under
Erasmus in this weekend’s World Cup final?

It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and
leaders that will call time on their international careers at week’s end.

The backlash and toxicity that followed their historic home series loss to Ireland was a humbling and public reminder of the expectation of being an All Black.

It’s why Saturday’s World Cup final is personal for the men wearing black.

All Blacks social media managers were forced to delete a thousand comments an hour following the humbling series loss in Wellington.
</item 7>

<item 8>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Ian Foster, who made two changes to his starting side and three in total, was forced to defend his centre’s actions while twisting the knife into Sexton’s
finish.

“It happens on most sporting parks when tensions are high and a lot at stake,” Foster said.

“Unfortunately recently it’s also included players saying a few things to referees. Is it right for the game? I don’t know. You don’t hear a lot of players
complaining about it.”

Advertisement

Captain Sam Cane added: “It’s players’ different personalities and emotions running high in those moments. There’s always people who have different tactics but
it’s always just left out there.”

Foster recalled Sam Whitelock to the starting side in the only change to the pack.

Whitelock, whose breakdown penalty win after 37 phases denied Ireland a match-winning penalty, has been recalled and will partner Scott Barrett in the
second-row. Brodie Retallick will come off the bench.
</item 8>

<item 9>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The All Blacks will experience a changing of the guard next season.

Indeed, the All Blacks’ social media channels described the team selection for the World Cup final as the “Last Dance”.

2015 World Cup winner Dane Coles will call time on his career, while his tight-five teammates Brodie Retallick and Sam Whitelock, who is bidding to become the
first player to win the Webb Ellis Cup for a third time, will head overseas.

Fellow World Cup winners Aaron Smith and Beauden Barrett will also head to Japan.

Coles, who was labelled by Samisoni Taukei’aho as a “Grandad” type figure in the All Blacks, has even been denied the chance of a farewell send-off after
missing selection.

Advertisement

“Yeah, tough,” said coach Ian Foster of the conversation with the 36-year-old around leaving him out.
</item 9>

<item 10>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“In the game in 2020 there were things you could pinpoint. You can’t say that one game means we can do it as there are a lot of games where we haven’t.

“That moment has to be just a part of it. What we have done since then, all the success and failures [matters in our preparation]. The other team are heavy
favourites, everyone is expecting a New Zealand and South Africa final. We just have to focus on what is required.”

Cheika has maintained that his Los Pumas side was too tight earlier in the tournament but has begun to free themselves up since qualifying for the knockout
stages – and rewarding their fans by doing so.

On inspiring the country?

“It’s not easy,” he said.

“We want to inspire not just by our results but by our ambition, by our ability to get back up and overcome obstacles.
</item 10>
=== END ITEM id=89402724fcd73d2d ===

=== ITEM id=7cc454c8d0642412 ===
Gold sentences and facts:
Sentence 0: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 0.0: Miami has five remaining opponents.
  Fact 0.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.
Sentence 1: 152.67 -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: Tyreek Hill missed last week's game.
  Fact 1.1: Tyreek Hill now needs to average almost 153 yards per game in his final three games.
  Fact 1.2: The figure cited is 152.67 yards per game.
  Fact 1.3: Tyreek Hill's stated goal is 2,000 receiving yards this season.

Retrieved items:

=== END ITEM id=7cc454c8d0642412 ===
